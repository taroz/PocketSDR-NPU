//
//  Pocket SDR - Accelerator Backend Plugin for OpenVINO (Intel NPU).
//
//  The parallel code search (carrier wipe-off, DFT, code multiplication, IDFT
//  and correlation power) of all Doppler bins is executed as one inference of
//  a static-shape tensor graph. The DFT of size M = A * B is decomposed into
//  dense complex matrix products (four-step DFT), because NPUs provide matrix
//  engines but no FFT primitive.
//
//    x[n1,n2] (n = B*n1+n2) -> carrier wipe-off (separable: a_k[n1]*b_k[n2])
//      -> F_A (left) -> twiddle W_M^(k1*n2) -> F_B (right) : X[k1+A*k2]
//      -> * code[k1+A*k2]
//      -> F_B^-1 (right) -> twiddle W_M^-(k1*p2) -> F_A^-1 (left) : c[B*p1+p2]
//      -> |c|^2
//
//  If the Doppler bins are spaced by integer multiples of the DFT bin (fs/M),
//  the forward DFT is computed once and shared (shift mode), since the carrier
//  wipe-off of bin k is a frequency shift of the DFT: X_k[m] = X_0[m+s_k]. In
//  the four-step order (m = k1+A*k2), the shift is a row gather from X_0
//  stacked with X_0 rolled by one column (carry of k1+s_k >= A). In shift mode,
//  the carrier wipe-off of the first bin is done by the host, because each
//  graph node costs a fixed NPU overhead (~30-90 us on Intel NPU) and the
//  separable carrier in the graph takes ~20 nodes.
//
//  The Doppler bins are processed in chunks of at most MAX_CHUNK / M bins per
//  inference. On Intel NPU, the execution time increases superlinearly with
//  the size of the intermediate tensors (K * M complex), e.g. M=128000 (4 ms
//  code): K=21 8.1 ms, K=81 81 ms.
//
//  Complex values are carried as real tensors (re/im halves stacked along the
//  contracted axis), so each complex matrix product is one real MatMul.
//  Intel NPU computes in FP16, so the DFTs are unitary and the inputs are
//  normalized to unit RMS to keep the values well inside the FP16 range. The
//  output powers are FP16 to halve the output tensor (NPU-to-host) size.
//
//  options: <device>[,direct] (OpenVINO device name: NPU (default),
//           CPU, ...)
//           direct: disable shift mode (for benchmark)
//           pipe  : pipelined chunks by 2 infer requests (faster for multiple
//                   chunks, but more CPU time by the memory contention with
//                   the NPU and the async execution)
//           q8, q16: integer emulation of code search (all MatMul inputs and
//                  constants quantized to 8 or 16 bits)
//           qa=b0-b1-b2-b3: bits of MatMul inputs of DFT (n1, n2) and IDFT
//                  (k2, k1) stages, qw=b: bits of constants, clip=k: clip
//                  range of quantized MatMul inputs (k sigma, default: 4)
//  environment: SDR_ACC_CACHE (compiled model cache dir, "": no cache)
//
//  Author:
//  T.SUZUKI
//
//  History:
//  2026-10-02  1.0  new
//  2026-10-02  1.1  host carrier wipe-off in shift mode, Doppler bin chunks
//  2026-10-03  1.2  carrier outer product by MatMul in direct mode, min
//                   Doppler bins per inference, integer quantization options,
//                   cache of code DFT, RMS of IF samples by subsamples
//
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <tuple>
#include <vector>
#include <openvino/openvino.hpp>
#include <openvino/opsets/opset13.hpp>
#include <openvino/core/preprocess/pre_post_process.hpp>
#if defined(__F16C__)
#include <immintrin.h>
#endif
#include "sdr_acc.h"

namespace o13 = ov::opset13;
typedef ov::Output<ov::Node> out_t;

#define MIN_FACT   8        // min DFT factor
#define MAX_FACT   2048     // max DFT factor (NPU memory of 2B x 2B matrix)
#define MAX_CHUNK  2688000  // max Doppler bins x DFT size per inference
#define MIN_CHUNK  8        // min Doppler bins per inference
#define CARR_BLK   256      // block size of host carrier wipe-off

struct model_t {            // compiled code search model
    int M, A, B, K, R, S;   // DFT size/factors, Doppler bins, output rows,
                            // Doppler bin step (DFT bins) (0: direct mode)
    int xi8;                // IF samples input as int8 (direct mode)
    ov::CompiledModel cm;   // compiled model
    ov::InferRequest rq[2]; // infer requests (pipelined chunks)
    std::mutex mtx;         // lock for infer requests
    const float *code[2] = {NULL, NULL}; // code DFT set in input tensors
    uint64_t code_sig[2] = {0, 0}; // signature of code DFT
    double s_c = 0.0;       // scale of code DFT
};

typedef std::tuple<int, int, int, int, int> model_key_t; // (M, K, R, S, xi8)

struct corr_model_t {       // compiled batched correlator model
    int N, C, n, B, L;      // samples, channels, rows/2, factors (N = B*L)
    int nch_prev;           // number of channels of previous call
    ov::CompiledModel cm;   // compiled model
    ov::InferRequest rq;    // infer request
    std::mutex mtx;         // lock for infer request
};

typedef std::tuple<int, int, int> corr_key_t; // (N, C, n)
static std::map<corr_key_t, std::unique_ptr<corr_model_t>> corr_models;

// global variables ------------------------------------------------------------
static ov::Core *core = NULL;
static std::string device = "NPU";
static int shift_mode = 1;
static int pipe_mode = 0;           // pipelined chunks (0:off,1:on)
static int qa[4] = {0};             // bits of quantized MatMul inputs (0: FP)
static int qw = 0;                  // bits of quantized constants (0: FP)
static double qclip = 4.0;          // clip range of quantized inputs (sigma)
static int nref = 0;                // reference count of init()
static std::map<model_key_t, std::unique_ptr<model_t>> models;
static std::mutex models_mtx;

// time (s) --------------------------------------------------------------------
static double now(void)
{
    return std::chrono::duration<double>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

// factorize DFT size (M = A * B, A <= B, A: max divisor <= sqrt(M)) -----------
static int factorize(int M, int *A, int *B)
{
    for (int a = (int)std::sqrt((double)M); a >= MIN_FACT; a--) {
        if (M % a) continue;
        *A = a;
        *B = M / a;
        return *B <= MAX_FACT;
    }
    return 0;
}

// constant tensor -------------------------------------------------------------
static out_t cst(const std::vector<float> &v, const ov::Shape &shape)
{
    return o13::Constant::create(ov::element::f32, shape, v);
}

// fake quantization to signed integer of bits (symmetric, range [-r,r]) ------
//   (integer emulation of the code search for NPUs without floating point:
//   the MatMul inputs and the constants are quantized, the MatMul outputs are
//   kept as the accumulators (int32))
static out_t fq(out_t t, double r, int bits)
{
    if (bits <= 0) return t;
    // q = clamp(round(t / step), -L, L) * step, L = 2^(bits-1) - 1
    double L = (double)((1 << (bits - 1)) - 1), step = r / L;
    auto c1 = o13::Constant::create(ov::element::f32, ov::Shape{}, {(float)(1.0 / step)});
    auto c2 = o13::Constant::create(ov::element::f32, ov::Shape{}, {(float)step});
    out_t q = std::make_shared<o13::Round>(std::make_shared<o13::Multiply>(t, c1),
        o13::Round::RoundMode::HALF_TO_EVEN);
    q = std::make_shared<o13::Clamp>(q, -L, L);
    return std::make_shared<o13::Multiply>(q, c2);
}

// quantize constant tensor by max abs value -----------------------------------
static out_t fqc(out_t t, int bits)
{
    if (bits <= 0) return t;
    auto c = std::dynamic_pointer_cast<o13::Constant>(t.get_node_shared_ptr());
    float m = 0.0f;
    for (float v : c->cast_vector<float>()) m = std::max(m, std::fabs(v));
    return fq(t, m, bits);
}

// quantize MatMul input of stage i (complex RMS 1 -> sigma 1/sqrt(2)) -------
//   (the code DFT is normalized by the max magnitude for quantization, so
//   |z * code| <= |z| and the same range bounds all stages)
static out_t fqa(out_t t, int i)
{
    return fq(t, qclip / std::sqrt(2.0), qa[i]);
}

// index constant --------------------------------------------------------------
static out_t idx(std::vector<int64_t> v)
{
    return o13::Constant::create(ov::element::i64, ov::Shape{v.size()}, v);
}

// complex multiply (ar + j ai) * (br + j bi) ----------------------------------
static void cmul(out_t ar, out_t ai, out_t br, out_t bi, out_t &cr, out_t &ci)
{
    cr = std::make_shared<o13::Subtract>(std::make_shared<o13::Multiply>(ar, br),
        std::make_shared<o13::Multiply>(ai, bi));
    ci = std::make_shared<o13::Add>(std::make_shared<o13::Multiply>(ar, bi),
        std::make_shared<o13::Multiply>(ai, br));
}

// split tensor into re/im halves along axis -----------------------------------
static void split(out_t t, int axis, out_t &re, out_t &im)
{
    auto s = std::make_shared<o13::Split>(t,
        o13::Constant::create(ov::element::i64, ov::Shape{}, {axis}), 2);
    re = s->output(0);
    im = s->output(1);
}

// split leading re/im axis of input tensor [2,...] ----------------------------
static void split0(out_t t, out_t &re, out_t &im)
{
    split(t, 0, re, im);
    re = std::make_shared<o13::Squeeze>(re, idx({0}));
    im = std::make_shared<o13::Squeeze>(im, idx({0}));
}

// DFT matrix as real block (left: [[R,-I],[I,R]], right: [[R,I],[-I,R]]) -------
// (rows: output rows of each re/im half of left matrix [2*rows,2n])
static out_t dft_mat(int n, int sgn, int left, int rows)
{
    std::vector<float> m(4 * rows * n);
    double sc = 1.0 / std::sqrt((double)n); // unitary
    for (int i = 0; i < rows; i++) for (int j = 0; j < n; j++) {
        double a = sgn * 2.0 * M_PI * (double)((int64_t)i * j % n) / n;
        float re = (float)(std::cos(a) * sc), im = (float)(std::sin(a) * sc);
        m[i*2*n+j] = re;
        m[(rows+i)*2*n+n+j] = re;
        m[i*2*n+n+j] = left ? -im : im;
        m[(rows+i)*2*n+j] = left ? im : -im;
    }
    return fqc(cst(m, ov::Shape{(size_t)2 * rows, (size_t)2 * n}), qw);
}

// slice tensor along axis [start,stop) ----------------------------------------
static out_t slice(out_t t, int axis, int start, int stop)
{
    return std::make_shared<o13::Slice>(t, idx({start}), idx({stop}), idx({1}),
        idx({axis}));
}

// shift DFT by Doppler bins (four-step order) [A,2B] -> [K,A,2B] --------------
static out_t shift_dft(out_t z, int A, int B, int K, int S)
{
    out_t zr, zi;
    split(z, 1, zr, zi);
    
    // X_0 rolled by one column (k2 -> k2+1) for rows with carry
    auto rr = std::make_shared<o13::Concat>(ov::OutputVector{slice(zr, 1, 1, B),
        slice(zr, 1, 0, 1)}, 1);
    auto ri = std::make_shared<o13::Concat>(ov::OutputVector{slice(zi, 1, 1, B),
        slice(zi, 1, 0, 1)}, 1);
    auto zz = std::make_shared<o13::Concat>(ov::OutputVector{z,
        std::make_shared<o13::Concat>(ov::OutputVector{rr, ri}, 1)}, 0);
    
    // gather rows k1+s_k of [X_0; rolled X_0] (s_k = k*S < A)
    std::vector<int32_t> ix(K * A);
    for (int k = 0; k < K; k++) for (int k1 = 0; k1 < A; k1++) {
        ix[k*A+k1] = k1 + k * S;
    }
    auto gi = o13::Constant::create(ov::element::i32,
        ov::Shape{(size_t)K, (size_t)A}, ix);
    return std::make_shared<o13::Gather>(zz, gi,
        o13::Constant::create(ov::element::i64, ov::Shape{}, {0}));
}

// twiddle factors W_M^(sgn*k1*n2) [A,B] ---------------------------------------
static void twiddle(int A, int B, int sgn, out_t &re, out_t &im)
{
    int64_t M = (int64_t)A * B;
    std::vector<float> r(A * B), i(A * B);
    for (int k1 = 0; k1 < A; k1++) for (int n2 = 0; n2 < B; n2++) {
        double a = sgn * 2.0 * M_PI * (double)((int64_t)k1 * n2 % M) / M;
        r[k1*B+n2] = (float)std::cos(a);
        i[k1*B+n2] = (float)std::sin(a);
    }
    re = fqc(cst(r, ov::Shape{(size_t)A, (size_t)B}), qw);
    im = fqc(cst(i, ov::Shape{(size_t)A, (size_t)B}), qw);
}

// build code search model -----------------------------------------------------
//   inputs : x [2,A,B], ca [2,K,A,1], cb [2,K,1,B], code [2,A,B] (direct mode)
//            x [2,A,B], code [2,A,B] (shift mode, x: carrier wiped)
//   output : P [K,R,B] (|corr|^2 of code phases n = B*p1+p2, p1 < R)
static std::shared_ptr<ov::Model> build_model(int A, int B, int K, int R,
    int S)
{
    size_t a = A, b = B, kc = S ? 1 : K;
    auto x  = std::make_shared<o13::Parameter>(ov::element::f32, ov::Shape{2, a, b});
    auto ca = std::make_shared<o13::Parameter>(ov::element::f32, ov::Shape{2, kc, a, 1});
    auto cb = std::make_shared<o13::Parameter>(ov::element::f32, ov::Shape{2, kc, 1, b});
    auto cc = std::make_shared<o13::Parameter>(ov::element::f32, ov::Shape{2, a, b});
    out_t xr, xi, ar, ai, br, bi, cr, ci, yr, yi, wr, wi, tr, ti, zr, zi, y1;
    
    if (S) { // x: carrier wiped by the first bin on host
        
        // DFT along n1 (left): [1,2A,B]
        auto y = std::make_shared<o13::Reshape>(x, idx({1, 2 * A, B}), false);
        y1 = std::make_shared<o13::MatMul>(dft_mat(A, -1, 1, A), fqa(y, 0));
    }
    else {
        // carrier a_k[n1] * b_k[n2] [Kc,A,B] by MatMul with inner dimension 2
        // (re: [ar,-ai] * [br;bi], im: [ar,ai] * [bi;br]), because the
        // broadcast complex multiply of the outer product is slow on Intel NPU
        split0(x, xr, xi);
        split0(ca, ar, ai);
        split0(cb, br, bi);
        auto l1 = std::make_shared<o13::Concat>(ov::OutputVector{ar,
            std::make_shared<o13::Negative>(ai)}, 2);
        auto l2 = std::make_shared<o13::Concat>(ov::OutputVector{ar, ai}, 2);
        auto r1 = std::make_shared<o13::Concat>(ov::OutputVector{br, bi}, 1);
        auto r2 = std::make_shared<o13::Concat>(ov::OutputVector{bi, br}, 1);
        cr = std::make_shared<o13::MatMul>(l1, r1);
        ci = std::make_shared<o13::MatMul>(l2, r2);
        
        // carrier wipe-off: y = x * carrier -> [Kc,A,B]
        cmul(xr, xi, cr, ci, yr, yi);
        
        // DFT along n1 (left): [Kc,2A,B]
        auto y = std::make_shared<o13::Concat>(ov::OutputVector{yr, yi}, 1);
        y1 = std::make_shared<o13::MatMul>(dft_mat(A, -1, 1, A), fqa(y, 0));
    }
    
    // twiddle and DFT along n2 (right): [Kc,A,2B]
    split(y1, 1, yr, yi);
    twiddle(A, B, -1, wr, wi);
    cmul(yr, yi, wr, wi, tr, ti);
    auto t = std::make_shared<o13::Concat>(ov::OutputVector{tr, ti}, 2);
    out_t z = std::make_shared<o13::MatMul>(fqa(t, 1), dft_mat(B, -1, 0, B));
    
    // shift DFT by Doppler bins: [K,A,2B]
    if (S) {
        z = shift_dft(std::make_shared<o13::Squeeze>(z, idx({0})), A, B, K, S);
    }
    // multiply code DFT
    split(z, 2, zr, zi);
    split0(fq(cc, 1.0, qw), cr, ci); // |code| <= 1 (normalized by max)
    cmul(zr, zi, cr, ci, tr, ti);
    
    // IDFT along k2 (right), twiddle and IDFT along k1 (left, R rows): [K,2R,B]
    auto p = std::make_shared<o13::Concat>(ov::OutputVector{tr, ti}, 2);
    out_t u = std::make_shared<o13::MatMul>(fqa(p, 2), dft_mat(B, 1, 0, B));
    split(u, 2, yr, yi);
    twiddle(A, B, 1, wr, wi);
    cmul(yr, yi, wr, wi, tr, ti);
    auto v = std::make_shared<o13::Concat>(ov::OutputVector{tr, ti}, 1);
    out_t c = std::make_shared<o13::MatMul>(dft_mat(A, 1, 1, R), fqa(v, 3));
    
    // correlation power [K,R,B]
    split(c, 1, yr, yi);
    out_t pw = std::make_shared<o13::Add>(std::make_shared<o13::Multiply>(yr, yr),
        std::make_shared<o13::Multiply>(yi, yi));
    if (S) {
        return std::make_shared<ov::Model>(ov::OutputVector{pw},
            ov::ParameterVector{x, cc}, "sdr_search_code");
    }
    return std::make_shared<ov::Model>(ov::OutputVector{pw},
        ov::ParameterVector{x, ca, cb, cc}, "sdr_search_code");
}

// get (or compile) model ------------------------------------------------------
static model_t *get_model(int M, int K, int R, int S, int A, int B, int xi8)
{
    std::lock_guard<std::mutex> lock(models_mtx);
    model_key_t key(M, K, R, S, xi8);
    auto it = models.find(key);
    if (it != models.end()) return it->second.get();

    std::unique_ptr<model_t> m(new model_t);
    m->M = M; m->A = A; m->B = B; m->K = K; m->R = R; m->S = S; m->xi8 = xi8;
    try {
        double t0 = now();
        auto model = build_model(A, B, K, R, S);
        ov::preprocess::PrePostProcessor ppp(model);
        ppp.output(0).tensor().set_element_type(ov::element::f16);
        if (xi8) { // int8 IF samples converted in model
            ppp.input(0).tensor().set_element_type(ov::element::i8);
        }
        ov::AnyMap cfg = {ov::hint::performance_mode(
            ov::hint::PerformanceMode::LATENCY)};
        m->cm = core->compile_model(ppp.build(), device, cfg);
        m->rq[0] = m->cm.create_infer_request();
        m->rq[1] = m->cm.create_infer_request();
        fprintf(stderr, "sdr_acc_ov: model compiled (%s M=%d(%dx%d) K=%d R=%d "
            "S=%d) %.2f s\n", device.c_str(), M, A, B, K, R, S, now() - t0);
    }
    catch (const std::exception &e) {
        fprintf(stderr, "sdr_acc_ov: model compile error: %s\n", e.what());
        models[key] = NULL; // not retried
        return NULL;
    }
    model_t *p = m.get();
    models[key] = std::move(m);
    return p;
}

// initialize backend ----------------------------------------------------------
static int ov_init(const char *opt)
{
    std::lock_guard<std::mutex> lock(models_mtx);
    if (core) { // already initialized (acquisition and tracking backends)
        nref++;
        return 1;
    }
    std::string o = opt ? opt : "";
    size_t p = o.find(',');
    for (std::string s = p != std::string::npos ? o.substr(p) : ""; !s.empty(); ) {
        size_t q = s.find(',', 1);
        std::string t = s.substr(1, q == std::string::npos ? q : q - 1);
        if (t == "direct") shift_mode = 0;
        else if (t == "pipe") pipe_mode = 1;
        else if (t == "q8" || t == "q16") { // all integer of bits
            int b = atoi(t.c_str() + 1);
            qa[0] = qa[1] = qa[2] = qa[3] = qw = b;
        }
        else if (t.compare(0, 3, "qa=") == 0) {
            sscanf(t.c_str() + 3, "%d-%d-%d-%d", qa, qa + 1, qa + 2, qa + 3);
        }
        else if (t.compare(0, 3, "qw=") == 0) qw = atoi(t.c_str() + 3);
        else if (t.compare(0, 5, "clip=") == 0) qclip = atof(t.c_str() + 5);
        else fprintf(stderr, "sdr_acc_ov: unknown option %s\n", t.c_str());
        s = q == std::string::npos ? "" : s.substr(q);
    }
    if (p != std::string::npos) o = o.substr(0, p);
    if (!o.empty()) device = o;
    try {
        core = new ov::Core();
        const char *env = getenv("SDR_ACC_CACHE"), *home = getenv("HOME");
        std::string dir = env ? env : (home ? std::string(home) +
            "/.cache/pocket_sdr/ov" : "");
        if (!dir.empty()) {
            core->set_property(ov::cache_dir(dir));
        }
        // check device availability
        (void)core->get_property(device, ov::device::full_name);
    }
    catch (const std::exception &e) {
        fprintf(stderr, "sdr_acc_ov: init error (%s): %s\n", device.c_str(),
            e.what());
        delete core;
        core = NULL;
        return 0;
    }
    nref = 1;
    return 1;
}

// finalize backend ------------------------------------------------------------
static void ov_exit(void)
{
    std::lock_guard<std::mutex> lock(models_mtx);
    if (--nref > 0) return;
    models.clear();
    corr_models.clear();
    delete core;
    core = NULL;
}

// Doppler bin step in DFT bins for shift mode (0: direct mode) ---------------
static int dop_step(int M, double fs, const double *fc, int K, int A)
{
    if (!shift_mode) return 0;
    if (K == 1) return 1;
    double s = (fc[1] - fc[0]) * M / fs;
    int S = (int)std::floor(s + 0.5);
    if (S < 1 || (int64_t)S * (K - 1) >= A) return 0;
    for (int k = 0; k < K; k++) {
        if (std::fabs((fc[k] - fc[0]) * M / fs - (double)S * k) > 1e-6) return 0;
    }
    return S;
}

// max magnitude of complex array ----------------------------------------------
static double max_abs(const float *x, int n)
{
    double m = 0.0;
    for (int i = 0; i < n; i++) {
        m = std::max(m, (double)x[i*2] * x[i*2] + (double)x[i*2+1] * x[i*2+1]);
    }
    return std::sqrt(m);
}

// RMS of complex array by subsamples (scale of FP16 range only) --------------
//   (the scale is exactly restored to the output powers, so an approximate
//   RMS by every RMS_STEP-th sample does not change the results)
#define RMS_STEP   16       // subsample step of RMS of IF samples

static double rms_sub(const float *x, int n)
{
    double s = 0.0;
    int m = 0;
    for (int i = 0; i < n; i += RMS_STEP, m++) {
        s += (double)x[i*2] * x[i*2] + (double)x[i*2+1] * x[i*2+1];
    }
    return m > 0 ? std::sqrt(s / m) : 0.0;
}

// RMS of int8 complex array by subsamples ------------------------------------
static double rms_sub_i8(const int8_t *x, int n)
{
    int64_t s = 0;
    int m = 0;
    for (int i = 0; i < n; i += RMS_STEP, m++) {
        s += x[i*2] * x[i*2] + x[i*2+1] * x[i*2+1];
    }
    return m > 0 ? std::sqrt((double)s / m) : 0.0;
}

// signature of code DFT (FNV-1a of strided samples) ---------------------------
static uint64_t code_sig(const float *code, int M)
{
    uint64_t h = 1469598103934665603ULL;
    int step = std::max(1, M / 256);
    for (int i = 0; i < 2 * M; i += 2 * step) {
        uint32_t v[2];
        std::memcpy(v, code + i, sizeof(v));
        h = (h ^ v[0]) * 1099511628211ULL;
        h = (h ^ v[1]) * 1099511628211ULL;
    }
    return h;
}

// RMS of complex array --------------------------------------------------------
static double rms(const float *x, int n)
{
    double s = 0.0;
    for (int i = 0; i < 2 * n; i++) s += (double)x[i] * x[i];
    return std::sqrt(s / n);
}

// accumulate FP16 powers with scale (p[n] += o[n] * scale) ---------------------
static void acc_f16(const ov::float16 *o, int n, float scale, float *p)
{
    int i = 0;
#if defined(__F16C__)
    __m256 ys = _mm256_set1_ps(scale);
    for ( ; i + 8 <= n; i += 8) {
        __m256 v = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(o + i)));
        _mm256_storeu_ps(p + i, _mm256_fmadd_ps(v, ys, _mm256_loadu_ps(p + i)));
    }
#endif
    for ( ; i < n; i++) p[i] += (float)o[i] * scale;
}

// fill input tensors of IF samples and carrier --------------------------------
static void set_input(model_t *m, ov::InferRequest &rq, const float *x,
    const int8_t *x8, double s_x, double fs, const double *fc, int set_x)
{
    int M = m->M, A = m->A, B = m->B, Kc = m->S ? 1 : m->K;
    float *tx = m->xi8 ? NULL : rq.get_input_tensor(0).data<float>();
    
    if (m->S) { // IF samples * carrier of fc[0] -> [re/im][n1][n2]
        // carrier = block phase exp(-j*w*n0) * rotation table exp(-j*w*i)
        float rr[CARR_BLK], ri[CARR_BLK];
        double w = -2.0 * M_PI * fc[0] / fs;
        for (int i = 0; i < CARR_BLK; i++) {
            rr[i] = (float)(std::cos(w * i) * s_x);
            ri[i] = (float)(std::sin(w * i) * s_x);
        }
        for (int n0 = 0; n0 < M; n0 += CARR_BLK) {
            double ph = std::fmod(w * n0, 2.0 * M_PI);
            float br = (float)std::cos(ph), bi = (float)std::sin(ph);
            int nb = std::min(CARR_BLK, M - n0);
            const float *xp = x + 2 * n0;
            float *yr = tx + n0, *yi = tx + M + n0;
            for (int i = 0; i < nb; i++) {
                float cr = br * rr[i] - bi * ri[i], ci = br * ri[i] + bi * rr[i];
                yr[i] = xp[i*2] * cr - xp[i*2+1] * ci;
                yi[i] = xp[i*2] * ci + xp[i*2+1] * cr;
            }
        }
        return;
    }
    // IF samples x[B*n1+n2] -> [re/im][n1][n2] (same for all chunks)
    // (int8: scale s_x is applied to the carrier a[n1])
    if (set_x && m->xi8) {
        int8_t *t8 = rq.get_input_tensor(0).data<int8_t>();
        for (int n = 0; n < M; n++) {
            t8[n    ] = x8[n*2  ];
            t8[n + M] = x8[n*2+1];
        }
    }
    for (int n = 0; set_x && !m->xi8 && n < M; n++) {
        tx[n    ] = (float)(x[n*2  ] * s_x);
        tx[n + M] = (float)(x[n*2+1] * s_x);
    }
    double sa = m->xi8 ? s_x : 1.0;
    // separable carrier exp(-j*2*pi*fc*(B*n1+n2)/fs) = a[n1] * b[n2]
    float *ta = rq.get_input_tensor(1).data<float>();
    float *tb = rq.get_input_tensor(2).data<float>();
    for (int k = 0; k < Kc; k++) {
        for (int n1 = 0; n1 < A; n1++) {
            double ph = fc[k] * B * n1 / fs, a = -2.0 * M_PI * (ph - std::floor(ph));
            ta[k*A+n1] = (float)(std::cos(a) * sa);
            ta[(Kc+k)*A+n1] = (float)(std::sin(a) * sa);
        }
        for (int n2 = 0; n2 < B; n2++) {
            double ph = fc[k] * n2 / fs, a = -2.0 * M_PI * (ph - std::floor(ph));
            tb[k*B+n2] = (float)std::cos(a);
            tb[(Kc+k)*B+n2] = (float)std::sin(a);
        }
    }
}

// parallel code search --------------------------------------------------------
static int search_code(const float *x, const int8_t *x8, const float *code,
    int M, double fs, const double *fc, int K, int nout, float *P, int ldP,
    sdr_acc_prof_t *prof)
{
    int A, B;
    if (!core || nout > M || !factorize(M, &A, &B)) return 0;
    
    // Doppler bins per inference (Kc) and number of chunks
    int Kmax = std::max(MIN_CHUNK, MAX_CHUNK / M), nch = (K + Kmax - 1) / Kmax;
    int Kc = (K + nch - 1) / nch;
    std::vector<double> fcc((size_t)Kc * nch);
    for (int k = 0; k < Kc * nch; k++) { // padded bins extrapolated
        fcc[k] = k < K ? fc[k] : fc[K-1] + (k - K + 1) * (fc[1] - fc[0]);
    }
    int R = (nout + B - 1) / B, S = dop_step(M, fs, fcc.data(), Kc, A);
    static thread_local std::vector<float> xf;
    if (x8 && S) { // shift mode: float IF samples for host carrier wipe-off
        xf.resize(2 * (size_t)M);
        for (int i = 0; i < 2 * M; i++) xf[i] = x8[i];
        x = xf.data();
        x8 = NULL;
    }
    model_t *m = get_model(M, Kc, R, S, A, B, x8 != NULL);
    if (!m) return 0;
    
    int quant = qw > 0 || qa[0] > 0 || qa[1] > 0 || qa[2] > 0 || qa[3] > 0;
    double s_x = x8 ? rms_sub_i8(x8, M) : rms_sub(x, M);
    if (s_x <= 0.0) return 0;
    s_x = 1.0 / s_x;
    uint64_t sig = code_sig(code, M);
    
    std::lock_guard<std::mutex> lock(m->mtx);
    double t_in = 0.0, t_exec = 0.0, t_out = 0.0, t0 = now();
    
    // code DFT in four-step output order: [k1][k2] = code[k1+A*k2]
    // (kept in the input tensors while the same code is searched)
    if (code != m->code[0] || sig != m->code_sig[0]) {
        double sc = quant ? max_abs(code, M) : rms(code, M);
        if (sc <= 0.0) return 0;
        m->s_c = 1.0 / sc;
        m->code[1] = NULL;
    }
    int nrq = pipe_mode && nch > 1 ? 2 : 1; // pipelined chunks
    for (int r = 0; r < nrq; r++) {
        if (code == m->code[r] && sig == m->code_sig[r]) continue;
        float *tc = m->rq[r].get_input_tensor(S ? 1 : 3).data<float>();
        for (int k1 = 0; k1 < A; k1++) for (int k2 = 0; k2 < B; k2++) {
            int i = k1 + A * k2;
            tc[k1*B+k2    ] = (float)(code[i*2  ] * m->s_c);
            tc[k1*B+k2 + M] = (float)(code[i*2+1] * m->s_c);
        }
        m->code[r] = code;
        m->code_sig[r] = sig;
    }
    // P += |corr|^2 (unitary DFT/IDFT -> CPU scale: 1 / (s_x * s_c * M)^2)
    double sc = 1.0 / (s_x * m->s_c * M);
    float scale = (float)(sc * sc);
    
    // chunks: input of chunk j and output of chunk j-1 are processed by host
    // while the NPU executes chunk j-1 and j
    double t_wait = 0.0;
    try {
        for (int j = 0; j <= nch; j++) {
            if (j < nch) {
                ov::InferRequest &rq = m->rq[j % nrq];
                double t1 = now();
                set_input(m, rq, x, x8, s_x, fs, fcc.data() + j * Kc, j < nrq);
                t_in += now() - t1;
                rq.start_async();
                if (nrq == 1) { // not pipelined: output of chunk j
                    double t2 = now();
                    rq.wait();
                    double t3 = now();
                    t_wait += t3 - t2;
                    const ov::float16 *out = rq.get_output_tensor(0).data<ov::float16>();
                    for (int k = 0; k < Kc && j * Kc + k < K; k++) {
                        acc_f16(out + (size_t)k * R * B, nout, scale,
                            P + (size_t)(j * Kc + k) * ldP);
                    }
                    t_out += now() - t3;
                }
            }
            if (j == 0 || nrq == 1) continue;
            ov::InferRequest &rq = m->rq[(j - 1) % nrq];
            double t2 = now();
            rq.wait();
            double t3 = now();
            t_wait += t3 - t2;
            const ov::float16 *out = rq.get_output_tensor(0).data<ov::float16>();
            for (int k = 0; k < Kc && (j - 1) * Kc + k < K; k++) {
                acc_f16(out + (size_t)k * R * B, nout, scale,
                    P + (size_t)((j - 1) * Kc + k) * ldP);
            }
            t_out += now() - t3;
        }
    }
    catch (const std::exception &e) {
        fprintf(stderr, "sdr_acc_ov: infer error: %s\n", e.what());
        return 0;
    }
    t_exec = now() - t0 - t_in - t_out; // execution not hidden by host
    if (prof) {
        int Ki = S ? 1 : Kc;
        prof->t_in = t_in;
        prof->t_exec = t_exec;
        prof->t_out = t_out;
        prof->bytes_in = 4.0 * (4.0 * M + (S ? 0.0 : 2.0 * Ki * (A + B))) * nch;
        prof->bytes_out = 2.0 * Kc * R * B * nch;
    }
    return 1;
}

// parallel code search (float IF samples) ------------------------------------
static int ov_search_code(const float *x, const float *code, int M, double fs,
    const double *fc, int K, int nout, float *P, int ldP, sdr_acc_prof_t *prof)
{
    return search_code(x, NULL, code, M, fs, fc, K, nout, P, ldP, prof);
}

// parallel code search (int8 IF samples) -------------------------------------
static int ov_search_code_i8(const int8_t *x, const float *code, int M,
    double fs, const double *fc, int K, int nout, float *P, int ldP,
    sdr_acc_prof_t *prof)
{
    return search_code(NULL, x, code, M, fs, fc, K, nout, P, ldP, prof);
}

//------------------------------------------------------------------------------
//  Batched tracking correlators
//
//  The correlators of the channels sharing the same IF samples are executed as
//  one inference. The carrier of channel c is separable for the sample index
//  s = L*b + t: carr_c[s] = a_c[b] * b_c[t] (computed by host in double). The
//  code replicas of the correlators are rotated by the host (memcpy of code
//  bank slices) and split before and after the code wrap-around (2 rows per
//  correlator). The correlations are computed by a batched MatMul:
//
//    z[c][s] = x[s] * a_c[b] * b_c[t]               [C,2,N]
//    out[c][r][re/im] = sum_s R[c][r][s] * z[c][s]  [C,2n,2]
//
//  The channels are padded to CORR_CBKT * 2^k to limit the number of compiled
//  models. The IF samples are scaled by 1/8 and the carrier by 1/scale of the
//  code to keep the correlations inside the FP16 range.
//------------------------------------------------------------------------------
#define CORR_CBKT  16       // channel bucket of batched correlator models
#define CORR_XSC   0.125    // scale of IF samples in batched correlator


// factorize correlation length (N = B * L, L: max divisor <= sqrt(N)) --------
static int factorize_corr(int N, int *B, int *L)
{
    for (int l = (int)std::sqrt((double)N); l >= 1; l--) {
        if (N % l) continue;
        *L = l;
        *B = N / l;
        return 1;
    }
    return 0;
}

// build batched correlator model ----------------------------------------------
//   inputs : x [2,N], ca [C,2,B,1], cb [C,2,1,L], R [C,2n,N] (int8)
//   output : corr [C,2n,2]
static std::shared_ptr<ov::Model> build_corr_model(int N, int C, int n, int B,
    int L)
{
    size_t c = C, b = B, l = L, nn = N, r = 2 * n;
    auto x  = std::make_shared<o13::Parameter>(ov::element::f32, ov::Shape{2, nn});
    auto ca = std::make_shared<o13::Parameter>(ov::element::f32, ov::Shape{c, 2, b, 1});
    auto cb = std::make_shared<o13::Parameter>(ov::element::f32, ov::Shape{c, 2, 1, l});
    auto R  = std::make_shared<o13::Parameter>(ov::element::f32, ov::Shape{c, r, nn});
    out_t ar, ai, br, bi, cr, ci, xr, xi, zr, zi;
    
    // carrier: a_c[b] * b_c[t] -> [C,1,N]
    split(ca, 1, ar, ai);
    split(cb, 1, br, bi);
    cmul(ar, ai, br, bi, cr, ci);
    cr = std::make_shared<o13::Reshape>(cr, idx({C, 1, N}), false);
    ci = std::make_shared<o13::Reshape>(ci, idx({C, 1, N}), false);
    
    // carrier wipe-off: z = x * carr -> [C,2,N]
    split(std::make_shared<o13::Reshape>(x, idx({1, 2, N}), false), 1, xr, xi);
    cmul(xr, xi, cr, ci, zr, zi);
    auto z = std::make_shared<o13::Concat>(ov::OutputVector{zr, zi}, 1);
    
    // correlations: R * z^T -> [C,2n,2]
    auto out = std::make_shared<o13::MatMul>(R, z, false, true);
    return std::make_shared<ov::Model>(ov::OutputVector{out},
        ov::ParameterVector{x, ca, cb, R}, "sdr_corr_batch");
}

// get (or compile) batched correlator model -----------------------------------
static corr_model_t *get_corr_model(int N, int C, int n)
{
    std::lock_guard<std::mutex> lock(models_mtx);
    corr_key_t key(N, C, n);
    auto it = corr_models.find(key);
    if (it != corr_models.end()) return it->second.get();
    
    std::unique_ptr<corr_model_t> m(new corr_model_t);
    m->N = N; m->C = C; m->n = n; m->nch_prev = C;
    if (!factorize_corr(N, &m->B, &m->L)) return NULL;
    try {
        double t0 = now();
        auto model = build_corr_model(N, C, n, m->B, m->L);
        ov::preprocess::PrePostProcessor ppp(model);
        for (int i = 0; i < 3; i++) {
            ppp.input(i).tensor().set_element_type(ov::element::f16);
        }
        ppp.input(3).tensor().set_element_type(ov::element::i8);
        ov::AnyMap cfg = {ov::hint::performance_mode(
            ov::hint::PerformanceMode::LATENCY)};
        m->cm = core->compile_model(ppp.build(), device, cfg);
        m->rq = m->cm.create_infer_request();
        std::memset(m->rq.get_input_tensor(3).data(), 0, (size_t)C * 2 * n * N);
        fprintf(stderr, "sdr_acc_ov: model compiled (%s corr N=%d(%dx%d) C=%d "
            "n=%d) %.2f s\n", device.c_str(), N, m->B, m->L, C, n, now() - t0);
    }
    catch (const std::exception &e) {
        fprintf(stderr, "sdr_acc_ov: model compile error: %s\n", e.what());
        corr_models[key] = NULL;
        return NULL;
    }
    corr_model_t *p = m.get();
    corr_models[key] = std::move(m);
    return p;
}

// carrier exp(-j*2*pi*ph) as FP16 --------------------------------------------
static void set_carr(double ph, double sc, ov::float16 *re, ov::float16 *im)
{
    double a = -2.0 * M_PI * (ph - std::floor(ph));
    *re = (ov::float16)(float)(std::cos(a) * sc);
    *im = (ov::float16)(float)(std::sin(a) * sc);
}

// batched tracking correlators ------------------------------------------------
static int ov_corr_batch(const int8_t *x, int N, double fs, int nch,
    sdr_acc_corr_t *corr, sdr_acc_prof_t *prof)
{
    if (!core || nch <= 0) return 0;
    int n = 0, C = CORR_CBKT;
    for (int i = 0; i < nch; i++) n = std::max(n, corr[i].n);
    while (C < nch) C *= 2;
    corr_model_t *m = get_corr_model(N, C, n);
    if (!m) return 0;
    
    std::lock_guard<std::mutex> lock(m->mtx);
    int B = m->B, L = m->L;
    double t0 = now();
    ov::float16 *tx = m->rq.get_input_tensor(0).data<ov::float16>();
    ov::float16 *ta = m->rq.get_input_tensor(1).data<ov::float16>();
    ov::float16 *tb = m->rq.get_input_tensor(2).data<ov::float16>();
    int8_t *tr = m->rq.get_input_tensor(3).data<int8_t>();
    
    // IF samples [I/Q][s]
    for (int s = 0; s < N; s++) {
        tx[s    ] = (ov::float16)(float)(x[s*2  ] * CORR_XSC);
        tx[s + N] = (ov::float16)(float)(x[s*2+1] * CORR_XSC);
    }
    for (int c = 0; c < nch; c++) {
        const sdr_acc_corr_t *p = corr + c;
        
        // separable carrier (code scale folded into a_c)
        double sc = 1.0 / std::max(1, (int)p->scale);
        for (int b = 0; b < B; b++) {
            set_carr(p->phi + p->fc * L * b / fs, sc, ta + (c*2)*B+b,
                ta + (c*2+1)*B+b);
        }
        for (int t = 0; t < L; t++) {
            set_carr(p->fc * t / fs, 1.0, tb + (c*2)*L+t, tb + (c*2+1)*L+t);
        }
        // code replicas before/after wrap-around
        for (int i = 0; i < n; i++) {
            int8_t *r1 = tr + ((size_t)c * 2 * n + 2 * i) * N, *r2 = r1 + N;
            if (i >= p->n) {
                std::memset(r1, 0, 2 * (size_t)N);
                continue;
            }
            int j = p->j[i];
            std::memcpy(r1, p->code[i] + N - j, j);
            std::memset(r1 + j, 0, N - j);
            std::memset(r2, 0, j);
            std::memcpy(r2 + j, p->code[i], N - j);
        }
    }
    for (int c = nch; c < m->nch_prev; c++) { // clear padded channels
        std::memset(tr + (size_t)c * 2 * n * N, 0, (size_t)2 * n * N);
    }
    m->nch_prev = nch;
    double t1 = now();
    try {
        m->rq.infer();
    }
    catch (const std::exception &e) {
        fprintf(stderr, "sdr_acc_ov: infer error: %s\n", e.what());
        return 0;
    }
    double t2 = now();
    const float *out = m->rq.get_output_tensor(0).data<float>();
    for (int c = 0; c < nch; c++) {
        for (int i = 0; i < corr[c].n; i++) {
            const float *o = out + ((size_t)c * 2 * n + 2 * i) * 2;
            for (int k = 0; k < 4; k++) {
                corr[c].corr[i*4+k] = (float)(o[k] * corr[c].scale / CORR_XSC);
            }
        }
    }
    if (prof) {
        prof->t_in = t1 - t0;
        prof->t_exec = t2 - t1;
        prof->t_out = now() - t2;
        prof->bytes_in = 2.0 * (2.0 * N + 2.0 * C * (B + L)) + 2.0 * C * n * N;
        prof->bytes_out = 4.0 * 4.0 * C * n;
    }
    return 1;
}

// plugin entry point ----------------------------------------------------------
extern "C" const sdr_acc_plugin_t *sdr_acc_plugin(void)
{
    static const sdr_acc_plugin_t plugin = {
        SDR_ACC_API_VER, "ov", ov_init, ov_exit, ov_search_code,
        ov_corr_batch, ov_search_code_i8
    };
    return &plugin;
}
