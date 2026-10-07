//
//  Pocket SDR - Accelerator Backend Plugin for Rockchip NPU (RKNN).
//
//  The parallel code search of the Doppler bins is executed on the RK3588 NPU
//  (FP16) by one of two implementations of the four-step DFT (M = A * B) of
//  the OpenVINO plugin (sdr_acc_ov.cpp). Both are the same algorithm as the CPU
//  reference (carrier wipe-off and DFT for each bin, no shared DFT).
//
//  (1) RKNN model (default): one inference per chunk of bins by a model
//  precompiled with RKNN-Toolkit2 (lib/build/gen_rknn_acq.py) for the shape
//  (M, Kc, R), searched in the order:
//    acq_M<M>_K<Kc>_R<R>_fq.rknn: INT8 of _f (option int8)
//    acq_M<M>_K<Kc>_R<R>_f.rknn: carrier folded into the DFT weights and the
//      twiddles (bins spaced by fs / M), input: IF samples wiped by the carrier
//      of the first bin of the chunk (host) (option nofold: not used)
//    acq_M<M>_K<Kc>_R<R>_w.rknn: carrier wipe-off in the model
//    acq_M<M>_K<Kc>_R<R>.rknn  : carrier wipe-off by the host (option hostwipe)
//  Each DFT stage is a 1x1 Conv over the channels (NCHW of the NPU) and the
//  axis changes are 2 Transposes (see gen_rknn_acq.py). The inputs and the
//  output are zero-copy buffers in the native layout of the NPU (NC1HWC2, FP16)
//  written and read by the host directly.
//
//  (2) RKNN matmul API (option mm, or no model file for the shape): the matmul
//  API has no element-wise operations, so each DFT stage is a separate NPU call
//  and the element-wise steps are done by the host (NEON) between the calls:
//
//    host: carrier wipe-off of x[n1,n2] (n = B*n1+n2)    -> [K][n2][n1]
//    NPU : x F_A                                         -> [K][n2][k1]
//    host: twiddle W_M^(k1*n2), transpose                 -> [K][k1][n2]
//    NPU : y F_B                         : X[k1+A*k2]    -> [K][k1][k2]
//    host: * code[k1+A*k2]                                -> [K][k1][k2]
//    NPU : z F_B^-1                                      -> [K][k1][p2]
//    host: twiddle W_M^-(k1*p2), transpose                -> [K][p2][k1]
//    NPU : w F_A^-1 (R columns)          : c[B*p1+p2]    -> [K][p2][p1]
//    host: P[k][B*p1+p2] += |c|^2
//
//  All products are "data x constant" (A: data rows, B: DFT matrix), so the
//  Doppler bins are stacked along the rows (M of the matmul) and the
//  transposes are fused into the twiddle steps. Complex values are
//  interleaved (re, im) along the rows and the constant real matrix has the
//  2x2 blocks [[re, im], [-im, re]], so a complex product is one real matmul.
//  The contracted dimension is padded to 32 and the output columns to 16
//  (alignment of the RK3588 matmul API).
//
//  The Doppler bins are split into chunks for the NPU cores (RK3588: 3). Each
//  chunk is processed by a worker thread with its own RKNN contexts bound to
//  one NPU core, so the host steps of a chunk overlap with the NPU execution
//  of the other chunks.
//
//  The DFTs are unitary and the inputs are normalized to unit RMS to keep the
//  values inside the FP16 range (the model output powers are scaled by 1/64).
//  The INT8 models (_fq) keep the final DFT (F_A^-1) and the powers in FP16.
//  Their power errors are larger than FP16 (NRMSE about 0.13 for a strong
//  signal vs 0.02) and the fixed quantization scales saturate by narrowband or
//  pulsed interference (CW J/N 10 dB: C/N0 metric +11 dB, false detections),
//  so they are not used by default.
//
//  options: [cores=n][,int8][,fp32][,mm][,hostwipe][,nofold][,model=dir]
//           cores=n: number of NPU cores (worker threads) (default: 3)
//           fp32   : FP32 output of matmul API (default: FP16)
//           mm     : matmul API only (no RKNN models)
//           hostwipe: carrier wipe-off by host (RKNN models acq_*.rknn)
//           nofold : no carrier folding (RKNN models acq_*_w.rknn with carrier
//                  wipe-off in the model)
//           int8   : INT8 models of carrier folding (acq_*_fq.rknn) if exist
//                  instead of FP16 models (acq_*_f.rknn) (weak to interference)
//           model=dir: directory of RKNN models (default: $SDR_ACC_RKNN_MODEL
//                  or <plugin dir>/rknn)
//  environment: SDR_ACC_RKNN_MODEL: directory of RKNN models
//               SDR_ACC_RKNN_PROF=1: print CPU time of the steps at exit
//
//  Author:
//  T.SUZUKI
//
//  History:
//  2026-10-07  1.0  new
//  2026-10-07  1.1  carrier folded models (_f)
//  2026-10-08  1.2  reset options by init, free RKNN models by exit,
//                   prefetch in output accumulation
//
#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <tuple>
#include <vector>
#include <arm_neon.h>
#include <dlfcn.h>
#include <rknn_api.h>
#include <rknn_matmul_api.h>
#include "sdr_acc.h"

#define MAX_CORE   3        // max number of NPU cores
#define MIN_FACT   8        // min factor of DFT size
#define MAX_KDIM   10240    // max contracted dimension of matmul
#define PI         3.1415926535897932

typedef __fp16 f16_t;

// matmul context --------------------------------------------------------------
struct mm_t {
    rknn_matmul_ctx ctx = 0;
    rknn_matmul_io_attr io;
    rknn_tensor_mem *A = NULL, *B = NULL, *C = NULL;
    int M = 0, K = 0, N = 0;
    ~mm_t() {
        if (!ctx) return;
        if (A) rknn_destroy_mem(ctx, A);
        if (B) rknn_destroy_mem(ctx, B);
        if (C) rknn_destroy_mem(ctx, C);
        rknn_matmul_destroy(ctx);
    }
};

// code search model of a worker (shape M, Kc, R) -----------------------------
struct model_t {
    int A, B, Kc, R;
    int K1, N1;             // padded dims of F_A step (2A)
    int K3, N3;             // padded dims of F_B steps (2B)
    int N7;                 // padded output dim of F_A^-1 step (2R)
    mm_t mm[4];             // F_A, F_B, F_B^-1, F_A^-1
};

// RKNN model of code search of a worker (shape M, Kc, R) ---------------------
//   inputs : xr, xi [Kc][A][1][B], cr, ci [1][B][1][A], output: P [Kc][R][1][B]
//   wipe-off in model: xr, xi [1][A][1][B], ar, ai [Kc][A][1][1],
//   br, bi [Kc][1][1][B] (carrier x * a[n1] * b[n2])
//   (native layout NC1HWC2 of FP16, zero-copy)
#define MAX_MIN    8        // max number of model inputs
#define RM_HOST    0        // model kind: carrier wipe-off by host
#define RM_WIPE    1        // model kind: carrier wipe-off in model (_w)
#define RM_FOLD    2        // model kind: carrier folded into DFT (_f)
#define RM_FOLDQ   3        // model kind: INT8 of RM_FOLD (_fq)
struct rmodel_t {
    rknn_context ctx = 0;
    int nin = 0;
    rknn_tensor_mem *in[MAX_MIN] = {NULL}, *out = NULL;
    rknn_tensor_attr ia[MAX_MIN], oa; // native attributes of inputs, output
    uint64_t code_gen = 0;      // generation of code DFT in inputs
    int kind = 0;               // model kind (RM_HOST, RM_WIPE, RM_FOLD)
    ~rmodel_t() {
        if (!ctx) return;
        for (int i = 0; i < MAX_MIN; i++) if (in[i]) rknn_destroy_mem(ctx, in[i]);
        if (out) rknn_destroy_mem(ctx, out);
        rknn_destroy(ctx);
    }
};

// DFT tables of a shape (shared by workers) -----------------------------------
struct table_t {
    std::vector<f16_t> t1;  // twiddle W_M^(k1*n2) [B][A] complex
    std::vector<f16_t> t2;  // twiddle W_M^-(k1*p2) [A][B] complex
};

// call of code search (shared by workers) -------------------------------------
struct call_t {
    int M, A, B, Kc, R, nout, ldP;
    double fs;
    const double *fc;       // carrier frequencies [Kc * nworker]
    int K;                  // number of valid bins
    const f16_t *xt;        // IF samples [2][B][A] (re/im, n2, n1), scaled
    const f16_t *ct;        // code DFT [A][B] complex (k1, k2), scaled
    uint64_t code_gen;      // generation of code DFT
    float cmax;             // max magnitude of code DFT ct
    int S;                  // Doppler bin step (DFT bins) (0: not uniform)
    const table_t *tab;
    float *P;
    float scale;
};

// worker thread ---------------------------------------------------------------
struct worker_t {
    int id = 0;
    std::thread th;
    std::map<std::tuple<int, int, int>, std::unique_ptr<model_t>> models;
    std::map<std::tuple<int, int, int, int>, std::unique_ptr<rmodel_t>> rmodels;
    std::vector<f16_t> ab;  // separable carrier of bin
    const call_t *call = NULL;
    int k0 = 0;             // first bin of chunk
    int ok = 0;
    double t[9] = {0};      // profile of steps (thread CPU time) (s)
};

static int ncore = MAX_CORE;
static int out_fp32 = 0;
static int use_model = 1;           // use RKNN models (0: matmul API only)
static int wipe_model = 1;          // carrier wipe-off in RKNN models
static int fold_model = 1;          // carrier folded into DFT in RKNN models
static int int8_model = 0;          // INT8 RKNN models (_fq)
static std::string model_dir;       // directory of RKNN models
static int nref = 0;
static int prof_on = 0;
static worker_t *workers = new worker_t[MAX_CORE]; // (never destroyed)
static std::map<int, std::unique_ptr<table_t>> tables;
// (never destroyed: exit() without rk_exit() must not wait for the workers)
static std::mutex &call_mtx = *new std::mutex;  // one code search at a time
static std::mutex &wk_mtx = *new std::mutex;
static std::condition_variable &wk_cv = *new std::condition_variable;
static std::condition_variable &done_cv = *new std::condition_variable;
static int wk_gen = 0, wk_pend = 0, wk_quit = 0, wk_nact = 0;
static uint64_t ncall = 0;

// current time (s) ------------------------------------------------------------
static double now(void)
{
    return std::chrono::duration<double>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

// thread CPU time (s) --------------------------------------------------------
static double cputime(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static int pad(int n, int a) { return (n + a - 1) / a * a; }

// factorize DFT size (M = A * B, A <= B, A: max divisor <= sqrt(M)) -----------
static int factorize(int M, int *A, int *B)
{
    for (int a = (int)std::sqrt((double)M); a >= MIN_FACT; a--) {
        if (M % a) continue;
        *A = a;
        *B = M / a;
        return 2 * *B <= MAX_KDIM;
    }
    return 0;
}

// DFT matrix as real matrix [K][N] (unitary, rows n < n_in, cols k < n_out) ----
//   F[2n][2k] = re, F[2n][2k+1] = im, F[2n+1][2k] = -im, F[2n+1][2k+1] = re
//   (w = exp(sgn * j*2*pi*n*k/len) / sqrt(len))
static std::vector<float> dft_mat(int len, int n_out, int sgn, int K, int N)
{
    std::vector<float> F((size_t)K * N, 0.0f);
    double s = 1.0 / std::sqrt((double)len);
    for (int n = 0; n < len; n++) for (int k = 0; k < n_out; k++) {
        double ph = sgn * 2.0 * PI * (double)((int64_t)n * k % len) / len;
        float re = (float)(std::cos(ph) * s), im = (float)(std::sin(ph) * s);
        F[(size_t)(2*n  )*N + 2*k] = re;
        F[(size_t)(2*n  )*N + 2*k+1] = im;
        F[(size_t)(2*n+1)*N + 2*k] = -im;
        F[(size_t)(2*n+1)*N + 2*k+1] = re;
    }
    return F;
}

// create matmul context: C[M][N] = A[M][K] * F[K][N] on NPU core -------------
static int mm_create(mm_t *m, int M, int K, int N, int core,
    const std::vector<float> &F)
{
    rknn_matmul_info info;
    memset(&info, 0, sizeof(info));
    memset(&m->io, 0, sizeof(m->io));
    info.M = M;
    info.K = K;
    info.N = N;
    info.type = out_fp32 ? RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32 :
        RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT16;
    info.B_layout = 1; // native
    info.AC_layout = 0;
    int ret = rknn_matmul_create(&m->ctx, &info, &m->io);
    if (ret < 0) {
        fprintf(stderr, "sdr_acc_rknn: matmul create error %d (M=%d K=%d N=%d)\n",
            ret, M, K, N);
        m->ctx = 0;
        return 0;
    }
    rknn_matmul_set_core_mask(m->ctx, (rknn_core_mask)(1 << core));
    m->A = rknn_create_mem(m->ctx, m->io.A.size);
    m->B = rknn_create_mem(m->ctx, m->io.B.size);
    m->C = rknn_create_mem(m->ctx, m->io.C.size);
    if (!m->A || !m->B || !m->C) return 0;
    memset(m->A->virt_addr, 0, m->io.A.size); // zero padding
    std::vector<f16_t> f(F.size());
    for (size_t i = 0; i < F.size(); i++) f[i] = (f16_t)F[i];
    rknn_matmul_info inf = info; // same B layout (FP16 output not supported)
    inf.type = RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32;
    rknn_B_normal_layout_to_native_layout(f.data(), m->B->virt_addr, K, N,
        &inf);
    if (rknn_matmul_set_io_mem(m->ctx, m->A, &m->io.A) < 0 ||
        rknn_matmul_set_io_mem(m->ctx, m->B, &m->io.B) < 0 ||
        rknn_matmul_set_io_mem(m->ctx, m->C, &m->io.C) < 0) {
        fprintf(stderr, "sdr_acc_rknn: matmul set io mem error\n");
        return 0;
    }
    m->M = M;
    m->K = K;
    m->N = N;
    return 1;
}

// get model of worker ---------------------------------------------------------
static model_t *get_model(worker_t *w, int M, int A, int B, int Kc, int R)
{
    auto key = std::make_tuple(M, Kc, R);
    auto it = w->models.find(key);
    if (it != w->models.end()) return it->second.get();

    std::unique_ptr<model_t> m(new model_t);
    m->A = A;
    m->B = B;
    m->Kc = Kc;
    m->R = R;
    m->K1 = pad(2 * A, 32);
    m->N1 = pad(2 * A, 16);
    m->K3 = pad(2 * B, 32);
    m->N3 = pad(2 * B, 16);
    m->N7 = pad(2 * R, 16);
    if (!mm_create(m->mm    , Kc * B, m->K1, m->N1, w->id,
            dft_mat(A, A, -1, m->K1, m->N1)) ||
        !mm_create(m->mm + 1, Kc * A, m->K3, m->N3, w->id,
            dft_mat(B, B, -1, m->K3, m->N3)) ||
        !mm_create(m->mm + 2, Kc * A, m->K3, m->N3, w->id,
            dft_mat(B, B, 1, m->K3, m->N3)) ||
        !mm_create(m->mm + 3, Kc * B, m->K1, m->N7, w->id,
            dft_mat(A, R, 1, m->K1, m->N7))) {
        return NULL;
    }
    model_t *p = m.get();
    w->models[key] = std::move(m);
    return p;
}

// get DFT tables of shape -----------------------------------------------------
static const table_t *get_table(int M, int A, int B)
{
    auto it = tables.find(M);
    if (it != tables.end()) return it->second.get();
    std::unique_ptr<table_t> t(new table_t);
    t->t1.resize(2 * (size_t)M);
    t->t2.resize(2 * (size_t)M);
    for (int n2 = 0; n2 < B; n2++) for (int k1 = 0; k1 < A; k1++) {
        double ph = 2.0 * PI * (double)((int64_t)n2 * k1 % M) / M;
        size_t i = (size_t)n2 * A + k1, j = (size_t)k1 * B + n2;
        t->t1[i*2] = (f16_t)std::cos(ph);
        t->t1[i*2+1] = (f16_t)-std::sin(ph);
        t->t2[j*2] = (f16_t)std::cos(ph);
        t->t2[j*2+1] = (f16_t)std::sin(ph);
    }
    const table_t *p = t.get();
    tables[M] = std::move(t);
    return p;
}

// output element of matmul as float -------------------------------------------
static inline void ld4(const void *C, size_t i, float32x4_t &re,
    float32x4_t &im)
{
    if (out_fp32) {
        float32x4x2_t v = vld2q_f32((const float *)C + i);
        re = v.val[0];
        im = v.val[1];
    }
    else {
        float16x4x2_t v = vld2_f16((const float16_t *)C + i);
        re = vcvt_f32_f16(v.val[0]);
        im = vcvt_f32_f16(v.val[1]);
    }
}

static inline void ld1(const void *C, size_t i, float &re, float &im)
{
    if (out_fp32) {
        re = ((const float *)C)[i];
        im = ((const float *)C)[i+1];
    }
    else {
        re = (float)((const f16_t *)C)[i];
        im = (float)((const f16_t *)C)[i+1];
    }
}

static inline void st4(f16_t *p, float32x4_t re, float32x4_t im)
{
    float16x4x2_t v = {{vcvt_f16_f32(re), vcvt_f16_f32(im)}};
    vst2_f16((float16_t *)p, v);
}

static inline void cmul4(float32x4_t ar, float32x4_t ai, float32x4_t br,
    float32x4_t bi, float32x4_t &cr, float32x4_t &ci)
{
    cr = vfmsq_f32(vmulq_f32(ar, br), ai, bi);
    ci = vfmaq_f32(vmulq_f32(ar, bi), ai, br);
}

// separable carrier of bin: a[n1] = exp(-j*2*pi*f*B*n1), b[n2] = exp(-j*2*pi*f*n2)
//   (f = fc / fs, de-interleaved FP16 a[2][A], b[2][B], phasor recurrence)
static void carr_sep(double f, int A, int B, f16_t *a, f16_t *b)
{
    for (int i = 0; i < 2; i++) {
        int n = i ? B : A;
        f16_t *p = i ? b : a;
        double ph = -2.0 * PI * std::fmod(f * (i ? 1 : B), 1.0);
        double dr = std::cos(ph), di = std::sin(ph), cr = 1.0, ci = 0.0;
        for (int j = 0; j < n; j++) {
            p[j] = (f16_t)cr;
            p[n+j] = (f16_t)ci;
            double t = cr * dr - ci * di;
            ci = cr * di + ci * dr;
            cr = t;
            if (j % 64 == 63) { // renormalize
                double s = 1.0 / std::sqrt(cr * cr + ci * ci);
                cr *= s;
                ci *= s;
            }
        }
    }
}

// carrier wipe-off of 8 samples: y = x[n1..n1+7] * a[n1..n1+7] * b ------------
static inline void wipe8(const f16_t *xr, const f16_t *xi, const f16_t *ar,
    const f16_t *ai, float16x8_t br, float16x8_t bi, float16x8_t &yr,
    float16x8_t &yi)
{
    float16x8_t vr = vld1q_f16((const float16_t *)ar);
    float16x8_t vi = vld1q_f16((const float16_t *)ai);
    float16x8_t cr = vfmsq_f16(vmulq_f16(vr, br), vi, bi);
    float16x8_t ci = vfmaq_f16(vmulq_f16(vr, bi), vi, br);
    float16x8_t x_r = vld1q_f16((const float16_t *)xr);
    float16x8_t x_i = vld1q_f16((const float16_t *)xi);
    yr = vfmsq_f16(vmulq_f16(x_r, cr), x_i, ci);
    yi = vfmaq_f16(vmulq_f16(x_r, ci), x_i, cr);
}

static inline void wipe1(const f16_t *xr, const f16_t *xi, const f16_t *ar,
    const f16_t *ai, float br, float bi, float &yr, float &yi)
{
    float cr = (float)*ar * br - (float)*ai * bi;
    float ci = (float)*ar * bi + (float)*ai * br;
    yr = (float)*xr * cr - (float)*xi * ci;
    yi = (float)*xr * ci + (float)*xi * cr;
}

// carrier wipe-off: A[(k,n2)][n1] = xt[n2][n1] * carr_k[B*n1+n2] -------------
static void step_carr(worker_t *w, const call_t *c, model_t *m)
{
    int A = c->A, B = c->B, M = A * B;
    f16_t *out = (f16_t *)m->mm[0].A->virt_addr;
    w->ab.resize(2 * (size_t)(A + B));
    f16_t *a = w->ab.data(), *b = a + 2 * A;
    for (int k = 0; k < c->Kc; k++) {
        carr_sep(c->fc[w->k0 + k] / c->fs, A, B, a, b);
        for (int n2 = 0; n2 < B; n2++) {
            const f16_t *xr = c->xt + (size_t)n2 * A, *xi = xr + M;
            f16_t *o = out + ((size_t)k * B + n2) * m->K1;
            float16x8_t br = vdupq_n_f16(b[n2]), bi = vdupq_n_f16(b[B+n2]);
            int n1 = 0;
            for ( ; n1 + 8 <= A; n1 += 8) {
                float16x8x2_t y;
                wipe8(xr + n1, xi + n1, a + n1, a + A + n1, br, bi, y.val[0],
                    y.val[1]);
                vst2q_f16((float16_t *)o + n1 * 2, y);
            }
            for ( ; n1 < A; n1++) {
                float yr, yi;
                wipe1(xr + n1, xi + n1, a + n1, a + A + n1, b[n2], b[B+n2], yr, yi);
                o[n1*2] = (f16_t)yr;
                o[n1*2+1] = (f16_t)yi;
            }
        }
    }
}

// transpose 4x4 (v[i][j] -> t[j][i]) ------------------------------------------
static inline void trans4(const float32x4_t *v, float32x4_t *t)
{
    float32x4x2_t a = vtrnq_f32(v[0], v[1]), b = vtrnq_f32(v[2], v[3]);
    t[0] = vcombine_f32(vget_low_f32(a.val[0]), vget_low_f32(b.val[0]));
    t[1] = vcombine_f32(vget_low_f32(a.val[1]), vget_low_f32(b.val[1]));
    t[2] = vcombine_f32(vget_high_f32(a.val[0]), vget_high_f32(b.val[0]));
    t[3] = vcombine_f32(vget_high_f32(a.val[1]), vget_high_f32(b.val[1]));
}

// twiddle and transpose: out[c][r] = in[r][c] * tw[r][c] ---------------------
//   in : [nr][ldi] complex interleaved (matmul output), tw: [nr][nc] complex
//   out: [nc][ldo] complex interleaved FP16
static void tw_trans(const void *in, int ldi, int nr, int nc, const f16_t *tw,
    f16_t *out, int ldo)
{
    int r = 0;
    if (nr % 4 == 0 && nc % 4 == 0) {
        for ( ; r < nr; r += 4) {
            for (int c = 0; c < nc; c += 4) {
                float32x4_t vr[4], vi[4];
                for (int i = 0; i < 4; i++) {
                    float32x4_t xr, xi;
                    ld4(in, (size_t)(r + i) * ldi + c * 2, xr, xi);
                    float16x4x2_t t = vld2_f16((const float16_t *)tw +
                        ((size_t)(r + i) * nc + c) * 2);
                    cmul4(xr, xi, vcvt_f32_f16(t.val[0]), vcvt_f32_f16(t.val[1]),
                        vr[i], vi[i]);
                }
                // transpose 4x4 (rows r+i, cols c+j) -> (rows c+j, cols r+i)
                float32x4_t tr[4], ti[4];
                trans4(vr, tr);
                trans4(vi, ti);
                for (int j = 0; j < 4; j++) {
                    st4(out + (size_t)(c + j) * ldo + r * 2, tr[j], ti[j]);
                }
            }
        }
        return;
    }
    for ( ; r < nr; r++) for (int c = 0; c < nc; c++) {
        float xr, xi;
        ld1(in, (size_t)r * ldi + c * 2, xr, xi);
        const f16_t *t = tw + ((size_t)r * nc + c) * 2;
        float tr = t[0], ti = t[1];
        out[(size_t)c * ldo + r * 2] = (f16_t)(xr * tr - xi * ti);
        out[(size_t)c * ldo + r * 2 + 1] = (f16_t)(xr * ti + xi * tr);
    }
}

// code multiplication: out[(k,k1)][k2] = in[(k,k1)][k2] * ct[k1][k2] ---------
static void step_code(const call_t *c, model_t *m)
{
    int A = c->A, B = c->B;
    const void *in = m->mm[1].C->virt_addr;
    f16_t *out = (f16_t *)m->mm[2].A->virt_addr;
    for (int k = 0; k < c->Kc; k++) for (int k1 = 0; k1 < A; k1++) {
        size_t row = (size_t)k * A + k1;
        const f16_t *ct = c->ct + (size_t)k1 * B * 2;
        f16_t *o = out + row * m->K3;
        int k2 = 0;
        for ( ; k2 + 4 <= B; k2 += 4) {
            float32x4_t xr, xi, yr, yi;
            ld4(in, row * m->N3 + k2 * 2, xr, xi);
            float16x4x2_t cv = vld2_f16((const float16_t *)ct + k2 * 2);
            cmul4(xr, xi, vcvt_f32_f16(cv.val[0]), vcvt_f32_f16(cv.val[1]), yr, yi);
            st4(o + k2 * 2, yr, yi);
        }
        for ( ; k2 < B; k2++) {
            float xr, xi;
            ld1(in, row * m->N3 + k2 * 2, xr, xi);
            float cr = ct[k2*2], ci = ct[k2*2+1];
            o[k2*2] = (f16_t)(xr * cr - xi * ci);
            o[k2*2+1] = (f16_t)(xr * ci + xi * cr);
        }
    }
}

// correlation power: P[k][B*p1+p2] += |c[(k,p2)][p1]|^2 * scale ---------------
static void step_power(worker_t *w, const call_t *c, model_t *m)
{
    int B = c->B, R0 = c->nout / B; // p1 < R0: all p2 valid
    const void *in = m->mm[3].C->virt_addr;
    float32x4_t vs = vdupq_n_f32(c->scale);
    for (int k = 0; k < c->Kc && w->k0 + k < c->K; k++) {
        float *P = c->P + (size_t)(w->k0 + k) * c->ldP;
        int p2 = 0, p1 = 0;
        if (B % 4 == 0) {
            for ( ; p2 < B; p2 += 4) {
                size_t row = (size_t)k * B + p2;
                for (p1 = 0; p1 + 4 <= R0; p1 += 4) {
                    float32x4_t v[4], t[4];
                    for (int i = 0; i < 4; i++) {
                        float32x4_t re, im;
                        ld4(in, (row + i) * m->N7 + p1 * 2, re, im);
                        v[i] = vfmaq_f32(vmulq_f32(re, re), im, im);
                    }
                    trans4(v, t);
                    for (int j = 0; j < 4; j++) {
                        float *p = P + (size_t)B * (p1 + j) + p2;
                        vst1q_f32(p, vfmaq_f32(vld1q_f32(p), t[j], vs));
                    }
                }
            }
            p2 = 0;
        }
        for ( ; p2 < B; p2++) {
            size_t row = (size_t)k * B + p2;
            for (int q = p1; q < c->R; q++) {
                int n = B * q + p2;
                if (n >= c->nout) break;
                float re, im;
                ld1(in, row * m->N7 + q * 2, re, im);
                P[n] += (re * re + im * im) * c->scale;
            }
        }
    }
}

// load RKNN model of worker (NULL: no model file or error) --------------------
static rmodel_t *get_rmodel(worker_t *w, int M, int Kc, int R, int kind)
{
    static const char *names[MAX_MIN] = {"xr", "xi", "cr", "ci", "ar", "ai",
        "br", "bi"};
    static const char *sfx[] = {"", "_w", "_f", "_fq"};
    auto key = std::make_tuple(M, Kc, R, kind);
    auto it = w->rmodels.find(key);
    if (it != w->rmodels.end()) return it->second.get(); // NULL: no model
    w->rmodels[key] = NULL;
    if (!use_model || model_dir.empty()) return NULL;
    
    char file[1024];
    snprintf(file, sizeof(file), "%s/acq_M%d_K%d_R%d%s.rknn", model_dir.c_str(),
        M, Kc, R, sfx[kind]);
    FILE *fp = fopen(file, "rb");
    if (!fp) {
        if (w->id == 0 && kind < RM_FOLD) {
            fprintf(stderr, "sdr_acc_rknn: no model %s (matmul API used)\n", file);
        }
        return NULL;
    }
    std::vector<uint8_t> buff;
    fseek(fp, 0, SEEK_END);
    buff.resize(ftell(fp));
    fseek(fp, 0, SEEK_SET);
    size_t n = fread(buff.data(), 1, buff.size(), fp);
    fclose(fp);
    std::unique_ptr<rmodel_t> m(new rmodel_t);
    rknn_input_output_num io;
    if (n != buff.size() ||
        rknn_init(&m->ctx, buff.data(), buff.size(), 0, NULL) < 0 ||
        rknn_set_core_mask(m->ctx, (rknn_core_mask)(1 << w->id)) < 0 ||
        rknn_query(m->ctx, RKNN_QUERY_IN_OUT_NUM, &io, sizeof(io)) < 0 ||
        io.n_input != (kind == RM_WIPE ? 8u : 4u) || io.n_output != 1) {
        fprintf(stderr, "sdr_acc_rknn: model load error %s\n", file);
        return NULL;
    }
    m->nin = io.n_input;
    m->kind = kind;
    for (int i = 0; i < m->nin; i++) {
        rknn_tensor_attr a;
        memset(&a, 0, sizeof(a));
        a.index = i;
        rknn_query(m->ctx, RKNN_QUERY_NATIVE_INPUT_ATTR, &a, sizeof(a));
        int j = 0;
        while (j < m->nin && strcmp(a.name, names[j])) j++;
        if (j < m->nin && a.n_dims == 4) {
            // native NHWC (C = 1: carrier b) -> normal layout FP16 NHWC
            // (converted by runtime, small)
            memset(&a, 0, sizeof(a));
            a.index = i;
            rknn_query(m->ctx, RKNN_QUERY_INPUT_ATTR, &a, sizeof(a));
            a.type = RKNN_TENSOR_FLOAT16;
            a.fmt = RKNN_TENSOR_NHWC;
            // (memory of padded W stride required by rknn_set_io_mem())
            int ws = std::max((int)a.w_stride, (int)a.dims[2]);
            a.size_with_stride = (uint32_t)a.dims[0] * a.dims[1] * ws * a.dims[3] * 2;
            a.w_stride = ws;
        }
        if (j >= m->nin || (a.type != RKNN_TENSOR_FLOAT16 &&
            a.type != RKNN_TENSOR_INT8) ||
            !(a.n_dims == 5 || (a.n_dims == 4 && a.fmt == RKNN_TENSOR_NHWC)) ||
            !(m->in[j] = rknn_create_mem(m->ctx, a.size_with_stride))) {
            fprintf(stderr, "sdr_acc_rknn: model input error %s\n", file);
            return NULL;
        }
        memset(m->in[j]->virt_addr, a.type == RKNN_TENSOR_INT8 ? a.zp : 0,
            a.size_with_stride);
        a.pass_through = a.n_dims == 5;
        if (rknn_set_io_mem(m->ctx, m->in[j], &a) < 0) return NULL;
        m->ia[j] = a;
    }
    memset(&m->oa, 0, sizeof(m->oa));
    m->oa.index = 0;
    rknn_query(m->ctx, RKNN_QUERY_NATIVE_OUTPUT_ATTR, &m->oa, sizeof(m->oa));
    if (m->oa.n_dims != 5 || (m->oa.type != RKNN_TENSOR_FLOAT16 &&
        m->oa.type != RKNN_TENSOR_INT8) ||
        !(m->out = rknn_create_mem(m->ctx, m->oa.size_with_stride)) ||
        rknn_set_io_mem(m->ctx, m->out, &m->oa) < 0) {
        fprintf(stderr, "sdr_acc_rknn: model output error %s\n", file);
        return NULL;
    }
    if (prof_on && w->id == 0) fprintf(stderr, "sdr_acc_rknn: model %s\n", file);
    rmodel_t *p = m.get();
    w->rmodels[key] = std::move(m);
    return p;
}

// offset of native (NC1HWC2) or normal (NHWC) tensor (H = 1) -----------------
static inline size_t nat_off(const rknn_tensor_attr &a, int n, int c, int x)
{
    if (a.n_dims == 4) { // NHWC [N][1][W][C]
        int WS = a.w_stride ? a.w_stride : a.dims[2];
        return ((size_t)n * WS + x) * a.dims[3] + c;
    }
    int C1 = a.dims[1], C2 = a.dims[4], WS = a.w_stride ? a.w_stride : a.dims[3];
    return (((size_t)n * C1 + c / C2) * WS + x) * C2 + c % C2;
}

// quantize to INT8 of native tensor (q = round(v / scale) + zp) ---------------
static inline int8x8_t quant8(float16x8_t v, float32x4_t is, float32x4_t zp)
{
    int32x4_t lo = vcvtnq_s32_f32(vfmaq_f32(zp, vcvt_f32_f16(vget_low_f16(v)), is));
    int32x4_t hi = vcvtnq_s32_f32(vfmaq_f32(zp, vcvt_f32_f16(vget_high_f16(v)), is));
    return vqmovn_s16(vcombine_s16(vqmovn_s32(lo), vqmovn_s32(hi)));
}

static inline int8_t quant1(float v, const rknn_tensor_attr &a)
{
    long q = std::lround(v / a.scale) + a.zp;
    return (int8_t)(q < -128 ? -128 : (q > 127 ? 127 : q));
}

// carrier wipe-off to native input: xr, xi [k][n1][0][n2] (k < nb) -----------
static void step_carr_nat(worker_t *w, const call_t *c, rmodel_t *m, int nb)
{
    int A = c->A, B = c->B, M = A * B;
    f16_t *yr = (f16_t *)m->in[0]->virt_addr, *yi = (f16_t *)m->in[1]->virt_addr;
    const rknn_tensor_attr &at = m->ia[0];
    int C2 = at.dims[4];
    w->ab.resize(2 * (size_t)(A + B));
    f16_t *a = w->ab.data(), *b = a + 2 * A;
    if (at.type == RKNN_TENSOR_INT8) { // INT8 input (xr, xi: same quantization)
        int8_t *qr = (int8_t *)yr, *qi = (int8_t *)yi;
        const rknn_tensor_attr &ai = m->ia[1];
        float32x4_t isr = vdupq_n_f32(1.0f / at.scale), zpr = vdupq_n_f32(at.zp);
        float32x4_t isi = vdupq_n_f32(1.0f / ai.scale), zpi = vdupq_n_f32(ai.zp);
        for (int k = 0; k < nb; k++) {
            carr_sep(c->fc[w->k0 + k] / c->fs, A, B, a, b);
            for (int n2 = 0; n2 < B; n2++) {
                const f16_t *xr = c->xt + (size_t)n2 * A, *xi = xr + M;
                float16x8_t br = vdupq_n_f16(b[n2]), bi = vdupq_n_f16(b[B+n2]);
                int n1 = 0;
                if (C2 % 8 == 0) {
                    for ( ; n1 + 8 <= A; n1 += 8) {
                        float16x8_t vr, vi;
                        wipe8(xr + n1, xi + n1, a + n1, a + A + n1, br, bi, vr, vi);
                        vst1_s8(qr + nat_off(at, k, n1, n2), quant8(vr, isr, zpr));
                        vst1_s8(qi + nat_off(ai, k, n1, n2), quant8(vi, isi, zpi));
                    }
                }
                for ( ; n1 < A; n1++) {
                    float vr, vi;
                    wipe1(xr + n1, xi + n1, a + n1, a + A + n1, b[n2], b[B+n2], vr, vi);
                    qr[nat_off(at, k, n1, n2)] = quant1(vr, at);
                    qi[nat_off(ai, k, n1, n2)] = quant1(vi, ai);
                }
            }
        }
        return;
    }
    for (int k = 0; k < nb; k++) {
        carr_sep(c->fc[w->k0 + k] / c->fs, A, B, a, b);
        for (int n2 = 0; n2 < B; n2++) {
            const f16_t *xr = c->xt + (size_t)n2 * A, *xi = xr + M;
            float16x8_t br = vdupq_n_f16(b[n2]), bi = vdupq_n_f16(b[B+n2]);
            int n1 = 0;
            if (C2 == 8) {
                for ( ; n1 + 8 <= A; n1 += 8) {
                    float16x8_t vr, vi;
                    wipe8(xr + n1, xi + n1, a + n1, a + A + n1, br, bi, vr, vi);
                    size_t o = nat_off(at, k, n1, n2);
                    vst1q_f16((float16_t *)yr + o, vr);
                    vst1q_f16((float16_t *)yi + o, vi);
                }
            }
            for ( ; n1 < A; n1++) {
                float vr, vi;
                wipe1(xr + n1, xi + n1, a + n1, a + A + n1, b[n2], b[B+n2], vr, vi);
                size_t o = nat_off(at, k, n1, n2);
                yr[o] = (f16_t)vr;
                yi[o] = (f16_t)vi;
            }
        }
    }
}

// IF samples and separable carrier to native inputs of wipe-off model ---------
//   xr, xi [0][n1][0][n2], ar, ai [k][n1][0][0], br, bi [k][0][0][n2]
static void step_wipe_nat(worker_t *w, const call_t *c, rmodel_t *m)
{
    int A = c->A, B = c->B, M = A * B;
    f16_t *p[MAX_MIN];
    for (int i = 0; i < m->nin; i++) p[i] = (f16_t *)m->in[i]->virt_addr;
    for (int n2 = 0; n2 < B; n2++) {
        const f16_t *xr = c->xt + (size_t)n2 * A, *xi = xr + M;
        int n1 = 0;
        if (m->ia[0].dims[4] == 8) {
            for ( ; n1 + 8 <= A; n1 += 8) {
                size_t o = nat_off(m->ia[0], 0, n1, n2);
                vst1q_f16((float16_t *)p[0] + o, vld1q_f16((const float16_t *)xr + n1));
                vst1q_f16((float16_t *)p[1] + o, vld1q_f16((const float16_t *)xi + n1));
            }
        }
        for ( ; n1 < A; n1++) {
            size_t o = nat_off(m->ia[0], 0, n1, n2);
            p[0][o] = xr[n1];
            p[1][o] = xi[n1];
        }
    }
    w->ab.resize(2 * (size_t)(A + B));
    f16_t *a = w->ab.data(), *b = a + 2 * A;
    for (int k = 0; k < c->Kc; k++) {
        carr_sep(c->fc[w->k0 + k] / c->fs, A, B, a, b);
        for (int n1 = 0; n1 < A; n1++) {
            size_t o = nat_off(m->ia[4], k, n1, 0);
            p[4][o] = a[n1];
            p[5][o] = a[A+n1];
        }
        for (int n2 = 0; n2 < B; n2++) {
            size_t o = nat_off(m->ia[6], k, 0, n2);
            p[6][o] = b[n2];
            p[7][o] = b[B+n2];
        }
    }
}

// code DFT to native input: cr, ci [0][k2][0][r*A+k1] (replicated along W) ---
//   (the first replica written and the others copied in the native layout)
static void step_code_nat(const call_t *c, rmodel_t *m)
{
    const rknn_tensor_attr &a = m->ia[2];
    int nrep = a.dims[3] / c->A, i8 = a.type == RKNN_TENSOR_INT8;
    size_t es = i8 ? 1 : 2;
    float s = i8 ? 1.0f / c->cmax : 1.0f; // INT8: normalized by max magnitude
    for (int j = 0; j < 2; j++) {
        const rknn_tensor_attr &aj = m->ia[2+j];
        uint8_t *p = (uint8_t *)m->in[2+j]->virt_addr;
        float is = i8 ? 1.0f / aj.scale : 1.0f;
        for (int k1 = 0; k1 < c->A; k1++) {
            const f16_t *ct = c->ct + (size_t)k1 * c->B * 2 + j;
            for (int k2 = 0; k2 < c->B; k2++) {
                size_t o = nat_off(aj, 0, k2, k1);
                float v = (float)ct[k2*2] * s;
                if (i8) {
                    int q = (int)lrintf(v * is) + aj.zp;
                    p[o] = (uint8_t)(int8_t)(q < -128 ? -128 : (q > 127 ? 127 : q));
                }
                else ((f16_t *)p)[o] = (f16_t)v;
            }
        }
        // copy replicas r = 1, ..., nrep-1 (W positions r*A, ..., r*A+A-1)
        int C1 = aj.dims[1], C2 = aj.dims[4], WS = aj.w_stride ? aj.w_stride : aj.dims[3];
        for (int c1 = 0; c1 < C1; c1++) {
            uint8_t *row = p + (size_t)c1 * WS * C2 * es;
            for (int r = 1; r < nrep; r++) {
                memcpy(row + (size_t)r * c->A * C2 * es, row, (size_t)c->A * C2 * es);
            }
        }
    }
    m->code_gen = c->code_gen;
}

// accumulate native output: P[k][B*p1+p2] += out[k][p1][0][p2] * scale --------
static void step_out_nat(worker_t *w, const call_t *c, rmodel_t *m, float scale)
{
    int B = c->B, C2 = m->oa.dims[4];
    const f16_t *out = (const f16_t *)m->out->virt_addr;
    float32x4_t vs = vdupq_n_f32(scale);
    for (int k = 0; k < c->Kc && w->k0 + k < c->K; k++) {
        float *P = c->P + (size_t)(w->k0 + k) * c->ldP;
        for (int p1 = 0; p1 < c->R; p1 += C2) {
            int p2 = 0;
            if (m->oa.type == RKNN_TENSOR_FLOAT16 && C2 == 8 && B % 4 == 0 &&
                p1 + 8 <= c->R &&
                B * (p1 + 8) <= c->nout) {
                for ( ; p2 < B; p2 += 4) {
                    float32x4_t lo[4], hi[4], t[4];
                    if ((p2 & 15) == 0) { // prefetch next cache lines (A55)
                        for (int i = 0; i < 8; i++) {
                            __builtin_prefetch(P + (size_t)B * (p1 + i) + p2 + 16, 1);
                        }
                        __builtin_prefetch((const float16_t *)out +
                            nat_off(m->oa, k, p1, p2) + 128, 0);
                    }
                    for (int j = 0; j < 4; j++) {
                        float16x8_t v = vld1q_f16((const float16_t *)out +
                            nat_off(m->oa, k, p1, p2 + j));
                        lo[j] = vcvt_f32_f16(vget_low_f16(v));
                        hi[j] = vcvt_f32_f16(vget_high_f16(v));
                    }
                    trans4(lo, t);
                    for (int i = 0; i < 4; i++) {
                        float *q = P + (size_t)B * (p1 + i) + p2;
                        vst1q_f32(q, vfmaq_f32(vld1q_f32(q), t[i], vs));
                    }
                    trans4(hi, t);
                    for (int i = 0; i < 4; i++) {
                        float *q = P + (size_t)B * (p1 + 4 + i) + p2;
                        vst1q_f32(q, vfmaq_f32(vld1q_f32(q), t[i], vs));
                    }
                }
                continue;
            }
            for (int i = 0; i < C2 && p1 + i < c->R; i++) {
                for (p2 = 0; p2 < B && B * (p1 + i) + p2 < c->nout; p2++) {
                    size_t o = nat_off(m->oa, k, p1 + i, p2);
                    float v = m->oa.type == RKNN_TENSOR_INT8 ?
                        (((const int8_t *)out)[o] - m->oa.zp) * m->oa.scale :
                        (float)out[o];
                    P[(size_t)B * (p1 + i) + p2] += v * scale;
                }
            }
        }
    }
}

// code search of a chunk by RKNN model -----------------------------------------
static int run_chunk_model(worker_t *w, const call_t *c, rmodel_t *m)
{
    double t0 = cputime(), t1;
    if (m->kind == RM_WIPE) step_wipe_nat(w, c, m);
    else step_carr_nat(w, c, m, m->kind >= RM_FOLD ? 1 : c->Kc);
    t1 = cputime(); w->t[0] += t1 - t0; t0 = t1;
    if (m->code_gen != c->code_gen) step_code_nat(c, m);
    t1 = cputime(); w->t[4] += t1 - t0; t0 = t1;
    if (rknn_run(m->ctx, NULL) < 0) return 0;
    t1 = cputime(); w->t[1] += t1 - t0; t0 = t1;
    // model output: power / 64 (INT8 model: code normalized by max magnitude)
    float sc = c->scale * 64.0f;
    if (m->kind == RM_FOLDQ) sc *= c->cmax * c->cmax;
    step_out_nat(w, c, m, sc);
    t1 = cputime(); w->t[8] += t1 - t0;
    return 1;
}

// code search of a chunk by worker --------------------------------------------
static int run_chunk(worker_t *w, const call_t *c)
{
    rmodel_t *rm = NULL;
    if (int8_model && fold_model && c->S == 1) {
        rm = get_rmodel(w, c->M, c->Kc, c->R, RM_FOLDQ);
    }
    if (!rm && fold_model && c->S == 1) {
        rm = get_rmodel(w, c->M, c->Kc, c->R, RM_FOLD);
    }
    if (!rm) rm = get_rmodel(w, c->M, c->Kc, c->R, wipe_model ? RM_WIPE : RM_HOST);
    if (rm) return run_chunk_model(w, c, rm);
    model_t *m = get_model(w, c->M, c->A, c->B, c->Kc, c->R);
    if (!m) return 0;
    int A = c->A, B = c->B;
    double t0 = cputime(), t1;

    step_carr(w, c, m);
    t1 = cputime(); w->t[0] += t1 - t0; t0 = t1;
    if (rknn_matmul_run(m->mm[0].ctx) < 0) return 0;
    t1 = cputime(); w->t[1] += t1 - t0; t0 = t1;
    for (int k = 0; k < c->Kc; k++) {
        size_t es = out_fp32 ? 4 : 2;
        tw_trans((const uint8_t *)m->mm[0].C->virt_addr + (size_t)k * B * m->N1 * es,
            m->N1, B, A, c->tab->t1.data(),
            (f16_t *)m->mm[1].A->virt_addr + (size_t)k * A * m->K3, m->K3);
    }
    t1 = cputime(); w->t[2] += t1 - t0; t0 = t1;
    if (rknn_matmul_run(m->mm[1].ctx) < 0) return 0;
    t1 = cputime(); w->t[3] += t1 - t0; t0 = t1;
    step_code(c, m);
    t1 = cputime(); w->t[4] += t1 - t0; t0 = t1;
    if (rknn_matmul_run(m->mm[2].ctx) < 0) return 0;
    t1 = cputime(); w->t[5] += t1 - t0; t0 = t1;
    for (int k = 0; k < c->Kc; k++) {
        size_t es = out_fp32 ? 4 : 2;
        tw_trans((const uint8_t *)m->mm[2].C->virt_addr + (size_t)k * A * m->N3 * es,
            m->N3, A, B, c->tab->t2.data(),
            (f16_t *)m->mm[3].A->virt_addr + (size_t)k * B * m->K1, m->K1);
    }
    t1 = cputime(); w->t[6] += t1 - t0; t0 = t1;
    if (rknn_matmul_run(m->mm[3].ctx) < 0) return 0;
    t1 = cputime(); w->t[7] += t1 - t0; t0 = t1;
    step_power(w, c, m);
    t1 = cputime(); w->t[8] += t1 - t0;
    return 1;
}

// worker thread function ------------------------------------------------------
static void worker_func(worker_t *w)
{
    int gen = 0;
    for (;;) {
        const call_t *c;
        {
            std::unique_lock<std::mutex> lock(wk_mtx);
            wk_cv.wait(lock, [&] { return wk_quit || wk_gen != gen; });
            if (wk_quit) return;
            gen = wk_gen;
            c = w->id < wk_nact ? w->call : NULL;
        }
        if (c) w->ok = run_chunk(w, c);
        std::lock_guard<std::mutex> lock(wk_mtx);
        if (--wk_pend == 0) done_cv.notify_one();
    }
}

// initialize backend ----------------------------------------------------------
static int rk_init(const char *opt)
{
    std::lock_guard<std::mutex> lock(call_mtx);
    if (nref++ > 0) return 1;
    ncore = MAX_CORE; // default options (also after rk_exit())
    out_fp32 = 0;
    use_model = wipe_model = fold_model = 1;
    int8_model = 0;
    model_dir.clear();
    std::string s = opt ? opt : "";
    for (size_t p = 0; p < s.size(); ) {
        size_t q = s.find(',', p);
        std::string t = s.substr(p, q == std::string::npos ? q : q - p);
        if (t.compare(0, 6, "cores=") == 0) {
            ncore = std::max(1, std::min(MAX_CORE, atoi(t.c_str() + 6)));
        }
        else if (t == "fp32") out_fp32 = 1;
        else if (t == "mm") use_model = 0;
        else if (t == "hostwipe") wipe_model = fold_model = 0;
        else if (t == "nofold") fold_model = 0;
        else if (t == "int8") int8_model = 1;
        else if (t.compare(0, 6, "model=") == 0) model_dir = t.substr(6);
        else if (!t.empty()) fprintf(stderr, "sdr_acc_rknn: unknown option %s\n",
            t.c_str());
        if (q == std::string::npos) break;
        p = q + 1;
    }
    const char *env = getenv("SDR_ACC_RKNN_PROF");
    prof_on = env && atoi(env);
    if (model_dir.empty()) { // model dir: $SDR_ACC_RKNN_MODEL or <plugin dir>/rknn
        Dl_info info;
        if ((env = getenv("SDR_ACC_RKNN_MODEL"))) model_dir = env;
        else if (dladdr((void *)rk_init, &info) && info.dli_fname) {
            std::string f = info.dli_fname;
            size_t q = f.rfind('/');
            model_dir = (q == std::string::npos ? std::string(".") :
                f.substr(0, q)) + "/rknn";
        }
    }

    // check NPU by a small matmul
    mm_t m;
    if (!mm_create(&m, 16, 32, 16, 0, std::vector<float>(32 * 16, 0.0f)) ||
        rknn_matmul_run(m.ctx) < 0) {
        fprintf(stderr, "sdr_acc_rknn: NPU not available\n");
        nref = 0;
        return 0;
    }
    wk_quit = wk_gen = wk_pend = 0;
    for (int i = 0; i < ncore; i++) {
        workers[i].id = i;
        workers[i].th = std::thread(worker_func, workers + i);
    }
    return 1;
}

// finalize backend ------------------------------------------------------------
static void rk_exit(void)
{
    std::lock_guard<std::mutex> lock(call_mtx);
    if (nref <= 0 || --nref > 0) return;
    {
        std::lock_guard<std::mutex> lk(wk_mtx);
        wk_quit = 1;
    }
    wk_cv.notify_all();
    static const char *name[9] = {"carr", "F_A", "tw1", "F_B", "code",
        "F_B^-1", "tw2", "F_A^-1", "power"};
    for (int i = 0; i < ncore; i++) {
        workers[i].th.join();
        if (prof_on && ncall > 0) {
            fprintf(stderr, "sdr_acc_rknn: worker %d (CPU ms/call):", i);
            for (int j = 0; j < 9; j++) {
                fprintf(stderr, " %s=%.3f", name[j], workers[i].t[j] * 1e3 / ncall);
            }
            fprintf(stderr, "\n");
        }
        workers[i].models.clear();
        workers[i].rmodels.clear();
        memset(workers[i].t, 0, sizeof(workers[i].t));
    }
    tables.clear();
    ncall = 0;
}

// Doppler bin step in DFT bins (fs / M) (0: not uniform integer step) ---------
static int dop_step(int M, double fs, const double *fc, int K)
{
    if (K == 1) return 1;
    double s = (fc[1] - fc[0]) * M / fs;
    int S = (int)std::floor(s + 0.5);
    if (S < 1) return 0;
    for (int k = 0; k < K; k++) {
        if (std::fabs((fc[k] - fc[0]) * M / fs - (double)S * k) > 1e-6) return 0;
    }
    return S;
}

// signature of code DFT (FNV-1a of strided samples) ---------------------------
static uint64_t code_sig(const float *code, int M)
{
    uint64_t h = 1469598103934665603ULL;
    int step = std::max(1, M / 256);
    for (int i = 0; i < 2 * M; i += 2 * step) {
        uint32_t v[2];
        memcpy(v, code + i, sizeof(v));
        h = (h ^ v[0]) * 1099511628211ULL;
        h = (h ^ v[1]) * 1099511628211ULL;
    }
    return h;
}

// parallel code search --------------------------------------------------------
static int search_code(const float *x, const int8_t *x8, const float *code,
    int M, double fs, const double *fc, int K, int nout, float *P, int ldP,
    sdr_acc_prof_t *prof)
{
    int A, B;
    if (nref <= 0 || nout > M || K < 1 || !factorize(M, &A, &B)) return 0;
    std::lock_guard<std::mutex> lock(call_mtx);
    double t0 = now();

    // Doppler bins per worker (padded bins extrapolated)
    int nw = std::min(ncore, K), Kc = (K + nw - 1) / nw;
    nw = (K + Kc - 1) / Kc;
    static std::vector<double> fcc;
    fcc.resize((size_t)Kc * nw);
    for (int k = 0; k < Kc * nw; k++) {
        fcc[k] = k < K ? fc[k] : fc[K-1] + (k - K + 1) *
            (K > 1 ? fc[1] - fc[0] : 0.0);
    }
    // IF samples [B][A] (n2, n1) and code DFT [A][B] (k1, k2) scaled to unit RMS
    static std::vector<f16_t> xt;
    static std::vector<f16_t> ct;
    xt.resize(2 * (size_t)M);
    ct.resize(2 * (size_t)M);
    double sx = 0.0;
    for (int i = 0; i < 2 * M; i++) {
        double v = x ? x[i] : x8[i];
        sx += v * v;
    }
    if (sx <= 0.0) return 0;
    float s_x = (float)std::sqrt(M / sx);
    for (int n1 = 0; n1 < A; n1++) for (int n2 = 0; n2 < B; n2++) {
        size_t i = (size_t)B * n1 + n2, j = (size_t)n2 * A + n1;
        xt[j] = (f16_t)((x ? x[i*2] : x8[i*2]) * s_x);
        xt[j+M] = (f16_t)((x ? x[i*2+1] : x8[i*2+1]) * s_x);
    }
    static const float *code_p = NULL;
    static uint64_t code_gen = 0, code_h = 0;
    static int code_M = 0;
    static float s_c = 0.0f, cmax = 1.0f;
    uint64_t hc = code_sig(code, M);
    if (code != code_p || hc != code_h || M != code_M) { // code DFT changed
        double sc = 0.0;
        for (int i = 0; i < 2 * M; i++) sc += (double)code[i] * code[i];
        if (sc <= 0.0) return 0;
        s_c = (float)std::sqrt(M / sc);
        for (int k1 = 0; k1 < A; k1++) for (int k2 = 0; k2 < B; k2++) {
            size_t i = (size_t)k1 + (size_t)A * k2, j = (size_t)k1 * B + k2;
            ct[j*2] = (f16_t)(code[i*2] * s_c);
            ct[j*2+1] = (f16_t)(code[i*2+1] * s_c);
        }
        cmax = 0.0f;
        for (int i = 0; i < M; i++) {
            cmax = std::max(cmax, std::hypot((float)ct[i*2], (float)ct[i*2+1]));
        }
        code_p = code;
        code_h = hc;
        code_M = M;
        code_gen++;
    }
    double s = 1.0 / ((double)s_x * s_c * M);
    call_t c = {M, A, B, Kc, (nout + B - 1) / B, nout, ldP, fs, fcc.data(),
        K, xt.data(), ct.data(), code_gen, cmax, dop_step(M, fs, fcc.data(),
        Kc * nw), get_table(M, A, B), P, (float)(s * s)};
    double t1 = now();

    // run workers
    {
        std::lock_guard<std::mutex> lk(wk_mtx);
        for (int i = 0; i < nw; i++) {
            workers[i].call = &c;
            workers[i].k0 = i * Kc;
            workers[i].ok = 0;
        }
        wk_nact = nw;
        wk_pend = ncore;
        wk_gen++;
    }
    wk_cv.notify_all();
    {
        std::unique_lock<std::mutex> lk(wk_mtx);
        done_cv.wait(lk, [] { return wk_pend == 0; });
    }
    int ok = 1;
    for (int i = 0; i < nw; i++) ok &= workers[i].ok;
    ncall++;
    if (prof) {
        prof->t_in = t1 - t0;
        prof->t_exec = now() - t1;
        prof->t_out = 0.0;
        prof->bytes_in = 2.0 * 2.0 * Kc * nw * M;
        prof->bytes_out = 2.0 * 2.0 * Kc * nw * c.R * B;
    }
    return ok;
}

// parallel code search (float IF samples) ------------------------------------
static int rk_search_code(const float *x, const float *code, int M,
    double fs, const double *fc, int K, int nout, float *P, int ldP,
    sdr_acc_prof_t *prof)
{
    return search_code(x, NULL, code, M, fs, fc, K, nout, P, ldP, prof);
}

// parallel code search (int8 IF samples) -------------------------------------
static int rk_search_code_i8(const int8_t *x, const float *code, int M,
    double fs, const double *fc, int K, int nout, float *P, int ldP,
    sdr_acc_prof_t *prof)
{
    return search_code(NULL, x, code, M, fs, fc, K, nout, P, ldP, prof);
}

// plugin entry point ----------------------------------------------------------
extern "C" const sdr_acc_plugin_t *sdr_acc_plugin(void)
{
    static const sdr_acc_plugin_t plugin = {
        SDR_ACC_API_VER, "rknn", rk_init, rk_exit, rk_search_code,
        NULL, rk_search_code_i8
    };
    return &plugin;
}
