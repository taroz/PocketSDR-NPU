//
//  Pocket SDR - Accelerator Backend Plugin for Hailo NPU (HailoRT).
//
//  The parallel code search of the Doppler bins is executed on a Hailo NPU
//  (e.g. Hailo-10H of Raspberry Pi AI HAT+ 2, Hailo-8 of AI HAT+) by a HEF
//  compiled with the Hailo Dataflow Compiler from the ONNX model of a carrier
//  folded four-step DFT (bins in the channels, batch 1, see
//  doc/npu_architecture.md):
//
//    host: carrier wipe-off of x by the first bin of the chunk, unit RMS
//          x[n1,n2] (n = B*n1+n2) -> NHWC [1][n2][n1] (float, quantized by
//          HailoRT), code DFT normalized by max magnitude [k1][k2] -> [1][k1][k2]
//    NPU : DFT_A (bins k folded), twiddles, transpose, DFT_B, * code, IDFT_B,
//          twiddles, transpose, IDFT_A (R rows, grouped by bin), power (16-bit)
//    host: P[k][B*p1+p2] += out[p2][k*R+p1] * scale
//
//  A HEF acq_M<M>_K<Kc>_R<R>_h.hef has Kc bins per inference (chunk) spaced by
//  fs / M. The chunks of a call are run asynchronously as frames of one job.
//  The INT8 model with the 16-bit power tail has the power NRMSE about 0.05-0.1
//  (20 ms non-coherent integration) and the C/N0 within 0.2 dB of the CPU
//  reference.
//
//  Notes on HEFs (Dataflow Compiler 5.4.0, HailoRT 5.4.0, Hailo-10H):
//  - Some multi-context HEFs output partly displaced samples on the device
//    (not in the emulator), e.g. Kc = 7, R = 100 (output features 700): +2 or
//    +B-6 samples in groups of 8 columns. R is padded to avoid it (Kc = 7,
//    R = 112 for M = 48000 and R = 64 for M = 16000 verified). A HEF has to
//    be verified on the device by the peak positions of all code offsets.
//  - M = 64000 (E1B/E1C at 8 MHz): Kc = 7 could not be allocated (transpose
//    of the format conversion), Kc = 3 outputs wrong powers with batch > 1,
//    Kc = 5 / R = 128 verified with batch 16.
//  - The inputs and outputs of the native order FCR of the device are
//    transferred at 400-800 MB/s, but those of the order NHWC (e.g. width not
//    a multiple of 8 in multi-context HEFs) are converted at 60-110 MB/s (e.g.
//    M = 320000 by A x B = 500 x 640: 199 fps, by 400 x 800: 402 fps, all
//    inputs FCR, L2CM 20 ms direct search 2291 -> 1260 ms). The order is
//    chosen by the compiler (not simply by the shape: 64000 by 200 x 320 or
//    160 x 400 and 16000 by 80 x 200 still have NHWC inputs). The
//    factorization with A, B multiples of 8 is tried first, then swapped (A >
//    B), then factorize() (A, B of the HEF by the input shape). Verified:
//    320000:1:200 (400 x 800), 64000:5:168 (320 x 200, all FCR, 816 -> 800
//    fps vs 576 fps of 64000:5:128 by 250 x 256, E1B 191 -> 157 ms); the
//    outputs of 64000:5:R by 200 x 320 (R = 100, 104, 108) and 320 x 200 (R =
//    160, 164, 172) are displaced.
//  - The calibration set has to include the codes of 10.23 Mcps (flat code
//    spectrum normalized by the max magnitude), otherwise the peaks of strong
//    signals are clipped and the Doppler bin of the max power is wrong.
//
//  - M = 16000: about 1200-1300 fps for any order or Kc = 1 (overhead of a
//    frame about 0.8 ms), 200 x 80 (all FCR) 1241 fps but displaced outputs
//    (R = 100); Kc = 11, 15, 21 not allocated in 40 min - 3 days.
//
//  Notes on HEFs (Dataflow Compiler 3.34.0, HailoRT 4.24.0, Hailo-8, -arch
//  hailo8 of gen_hailo_acq.py, HEFs in ../linux/hailo8 by option hef=):
//  - Only single-context HEFs are allocated (about Kc x M <= 150000 and Kc x A
//    <= 900): multi-context ones fail (16-bit or wide element-wise layers not
//    supported, partition timeout) incl. the long codes (M >= 160000).
//  - Verified on the device by the peak positions of L1CA (L5I does not show
//    the displaced outputs): 16000:7:48 (80 x 200), 48000:3:88 (160 x 300),
//    64000:2:88 (160 x 400, E1B). Displaced: 48000:3 by 200 x 240 (R = 112),
//    150 x 320, 120 x 400, 96 x 500, 80 x 600, 64000:2 by 80 x 800.
//
//  All windows of an acquisition are searched in one call by
//  search_code_i8_multi() (the frames of the windows and chunks queued at
//  once), which is much faster with the batch size 8-16 than the call per
//  window. The IF samples are wiped by the carrier of the first bin once per
//  window and the code DFT is rolled by the first bin of each chunk (shared DFT
//  identity), which reduces the host CPU time (e.g. L1CP 10 ms direct search:
//  input preparation 161 -> 16 ms per window).
//
//  options: [hef=dir][,batch=n]
//           batch=n: batch size of the models (default: HailoRT default,
//                  8-16 recommended for multi-window search)
//           hef=dir: directory of HEFs (default: $SDR_ACC_HAILO_HEF or
//                  <plugin dir>/hailo)
//  environment: SDR_ACC_HAILO_HEF: directory of HEFs
//
//  Author:
//  T.SUZUKI
//
//  History:
//  2026-10-11  1.0  new
//  2026-10-11  1.1  add model of Kc = 5
//  2026-10-12  1.2  code DFT rolled by chunk instead of the carrier wipe-off
//                   of the IF samples per frame (shared DFT identity)
//  2026-10-13  1.3  factorization of the HEF by the input shape, A, B
//                   multiples of 8 (and swapped) tried first (native input
//                   order)
//  2026-10-13  1.4  outputs accumulated by chunk overlapped with the
//                   inference (transposed by tiles, bins by threads), model
//                   of the fewest chunks (Kc up to 21)
//  2026-10-14  1.5  model selected from the HEFs in hef_dir (any Kc, any
//                   factorization A x B), Hailo-8 (HailoRT 4.24)
//  2026-10-14  1.6  shared DFT models (_hs, X by FFTW), code offset 0
//                   replaced by the neighbors (bias of the quantization)
//
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <dlfcn.h>
#include <dirent.h>
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <mutex>
#include <chrono>
#include <algorithm>
#include <thread>
#include <atomic>
#include <array>
#include <fftw3.h>
#include <hailo/hailort.hpp>
#include "sdr_acc.h"

using namespace hailort;
using Bindings = ConfiguredInferModel::Bindings;

#define PI         3.1415926535897932
#define TIMEOUT_MS 10000    // inference timeout (ms)

// HEF model -------------------------------------------------------------------
struct hmodel_t {
    std::shared_ptr<InferModel> im;
    std::unique_ptr<ConfiguredInferModel> cim;
    std::string in_name[4];             // xr, xi, cr, ci
    std::string out_name;
    int M = 0, Kc = 0, R = 0, A = 0, B = 0, sh = 0; // sh: shared DFT model
    float qs[4], qz[4], os = 1.0f, oz = 0.0f; // quantization of inputs/output
    std::vector<std::vector<uint8_t>> xr, xi;  // IF samples per window
    std::vector<uint8_t> cr, ci;               // code DFT [k1][k2]
    std::vector<std::vector<uint8_t>> crs, cis; // rolled code DFT (ring)
    std::vector<std::vector<uint16_t>> out;    // output per frame
    std::vector<std::unique_ptr<std::atomic<uint32_t>[]>> done; // done flags
    int ndone = 0;                             // (never freed for late callbacks)
    uint32_t gen = 0;                          // generation of the call
};
static int batch_size = 0;              // batch size (0: HailoRT default)

static std::unique_ptr<VDevice> vdev;
static std::map<std::string, std::unique_ptr<hmodel_t>> hmodels;
static std::string hef_dir;
static std::vector<std::array<int, 4>> hefs; // HEFs in hef_dir (M, Kc, R, sh)
static std::mutex call_mtx;

// current time (s) ------------------------------------------------------------
static double now(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

// DFT size factorization M = A * B (same as gen_rknn_acq.py) ------------------
static int factorize(int M, int *A, int *B)
{
    int best = 0;
    for (int a = 8; a * a <= M; a++) {
        if (M % a || (M / a) % 4) continue;
        best = a;
    }
    if (!best) return 0;
    *A = best; *B = M / best;
    return 1;
}

// DFT size factorization with A, B multiples of 8 (inputs in the native order
// FCR of the device: NHWC inputs are converted at 60-110 MB/s) ----------------
static int factorize8(int M, int *A, int *B)
{
    int best = 0;
    for (int a = 8; a * a <= M; a += 8) {
        if (M % a || (M / a) % 8) continue;
        best = a;
    }
    if (!best) return 0;
    *A = best; *B = M / best;
    return 1;
}

// load HEF model for (M, Kc, R, sh) -------------------------------------------
static hmodel_t *get_model(int M, int Kc, int R, int sh)
{
    char key[64];
    snprintf(key, sizeof(key), "acq_M%d_K%d_R%d_h%s", M, Kc, R, sh ? "s" : "");
    auto it = hmodels.find(key);
    if (it != hmodels.end()) return it->second.get();
    std::string file = hef_dir + "/" + key + ".hef";
    FILE *fp = fopen(file.c_str(), "rb");
    if (!fp) {
        hmodels[key] = nullptr;
        return NULL;
    }
    fclose(fp);
    std::unique_ptr<hmodel_t> m(new hmodel_t);
    auto im = vdev->create_infer_model(file);
    if (!im) {
        fprintf(stderr, "sdr_acc_hailo: model load error %s\n", file.c_str());
        hmodels[key] = nullptr;
        return NULL;
    }
    m->im = im.release();
    auto &ins = m->im->inputs();
    if (ins.size() != 4 || m->im->outputs().size() != 1) {
        fprintf(stderr, "sdr_acc_hailo: model I/O error %s\n", file.c_str());
        hmodels[key] = nullptr;
        return NULL;
    }
    // inputs input_layer1-4 = xr, xi, cr, ci (order of the ONNX inputs)
    std::vector<std::string> names;
    for (auto &s : ins) names.push_back(s.name());
    std::sort(names.begin(), names.end());
    for (int i = 0; i < 4; i++) {
        m->in_name[i] = names[i];
        auto in = m->im->input(names[i]);
        in->set_format_type(HAILO_FORMAT_TYPE_UINT8);
        auto q = in->get_quant_infos()[0];
        m->qs[i] = q.qp_scale; m->qz[i] = q.qp_zp;
    }
    m->out_name = m->im->outputs()[0].name();
    auto out = m->im->output();
    out->set_format_type(HAILO_FORMAT_TYPE_UINT16);
    out->set_format_order(HAILO_FORMAT_ORDER_NHWC);
    auto q = out->get_quant_infos()[0];
    m->os = q.qp_scale; m->oz = q.qp_zp;
    if (batch_size > 0) m->im->set_batch_size(batch_size);
    auto cim = m->im->configure();
    if (!cim) {
        fprintf(stderr, "sdr_acc_hailo: model configure error %s\n", file.c_str());
        hmodels[key] = nullptr;
        return NULL;
    }
    m->cim.reset(new ConfiguredInferModel(cim.release()));
    m->M = M; m->Kc = Kc; m->R = R; m->sh = sh;
    auto s = m->im->input(names[0])->shape(); // xr [1][B][A] (H, W, features)
    m->A = (int)(sh ? s.width : s.features); // sh: X [1][A][B]
    m->B = (int)(sh ? s.features : s.width);
    if (m->A * m->B != M) {
        fprintf(stderr, "sdr_acc_hailo: model shape error %s\n", file.c_str());
        hmodels[key] = nullptr;
        return NULL;
    }
    hmodel_t *p = m.get();
    hmodels[key] = std::move(m);
    return p;
}

// carrier table exp(-j*2*pi*f*n) (n = 0..M-1) -----------------------------
static void carr_table(double f, int M, int B, float *tab)
{
    double dr = cos(-2.0 * PI * f), di = sin(-2.0 * PI * f);
    for (int n1 = 0; n1 < M / B; n1++) { // renormalized every row
        double ph = -2.0 * PI * fmod(f * B * n1, 1.0), cr = cos(ph), ci = sin(ph);
        for (int n2 = 0; n2 < B; n2++) {
            int n = B * n1 + n2;
            tab[2*n] = (float)cr; tab[2*n+1] = (float)ci;
            double t = cr * dr - ci * di;
            ci = cr * di + ci * dr; cr = t;
        }
    }
}

// quantize IF samples of a window wiped by a carrier table (NHWC [n2][n1]) ----
//   (transposed by tiles of TILE rows n1 for the cache)
#define TILE 16     // tile rows of the input transpose
#define ATILE 32    // tile rows of the output transpose
template <typename T>
static void prep_frame_t(int A, int B, const T *__restrict x,
    const float *__restrict tab, float sr, float si, float zr, float zi,
    uint8_t *__restrict xr, uint8_t *__restrict xi)
{
    for (int n1b = 0; n1b < A; n1b += TILE) {
        int n1e = std::min(A, n1b + TILE);
        for (int n2 = 0; n2 < B; n2++) {
            for (int n1 = n1b; n1 < n1e; n1++) {
                int n = B * n1 + n2;
                float re = x[2*n], im = x[2*n+1], cr = tab[2*n], ci = tab[2*n+1];
                float q1 = (re * cr - im * ci) * sr + zr, q2 = (re * ci + im * cr) * si + zi;
                xr[n2 * A + n1] = (uint8_t)(q1 < 0.f ? 0.f : (q1 > 255.f ? 255.f : q1));
                xi[n2 * A + n1] = (uint8_t)(q2 < 0.f ? 0.f : (q2 > 255.f ? 255.f : q2));
            }
        }
    }
}

static void prep_frame(const hmodel_t *m, const float *x, const int8_t *x8,
    const float *tab, double sx, uint8_t *xr, uint8_t *xi)
{
    float sr = (float)(1.0 / sx / m->qs[0]), si = (float)(1.0 / sx / m->qs[1]);
    float zr = m->qz[0] + 0.5f, zi = m->qz[1] + 0.5f;
    if (x) prep_frame_t(m->A, m->B, x, tab, sr, si, zr, zi, xr, xi);
    else prep_frame_t(m->A, m->B, x8, tab, sr, si, zr, zi, xr, xi);
}

// quantize the unitary DFT of IF samples of a window wiped by a carrier table --
//   (shared DFT model: X [k1][k2] of X[A*k2+k1], FFT by FFTW, buffers of M)
static std::map<int, fftwf_plan> fplans; // FFTW plans by M (by call_mtx)

static fftwf_plan get_plan(int M)
{
    auto it = fplans.find(M);
    if (it != fplans.end()) return it->second;
    fftwf_complex *a = fftwf_alloc_complex(M), *b = fftwf_alloc_complex(M);
    fftwf_plan p = fftwf_plan_dft_1d(M, a, b, FFTW_FORWARD, FFTW_ESTIMATE);
    fftwf_free(a); fftwf_free(b);
    return fplans[M] = p;
}

template <typename T>
static void prep_frame_sh(const hmodel_t *m, const T *__restrict x,
    const float *__restrict tab, double sx, fftwf_complex *in, fftwf_complex *out,
    uint8_t *__restrict xr, uint8_t *__restrict xi)
{
    int M = m->M, A = m->A, B = m->B;
    for (int n = 0; n < M; n++) {
        float re = x[2*n], im = x[2*n+1], cr = tab[2*n], ci = tab[2*n+1];
        in[n][0] = re * cr - im * ci; in[n][1] = re * ci + im * cr;
    }
    fftwf_execute_dft(get_plan(M), in, out);
    float sr = (float)(1.0 / sqrt((double)M) / sx / m->qs[0]);
    float si = (float)(1.0 / sqrt((double)M) / sx / m->qs[1]);
    float zr = m->qz[0] + 0.5f, zi = m->qz[1] + 0.5f;
    for (int k2 = 0; k2 < B; k2++) for (int k1 = 0; k1 < A; k1++) {
        const float *o = out[A * k2 + k1];
        float q1 = o[0] * sr + zr, q2 = o[1] * si + zi;
        xr[k1 * B + k2] = (uint8_t)(q1 < 0.f ? 0.f : (q1 > 255.f ? 255.f : q1));
        xi[k1 * B + k2] = (uint8_t)(q2 < 0.f ? 0.f : (q2 > 255.f ? 255.f : q2));
    }
}

// parallel code search of NW windows (non-coherent integration) -------------
//   the IF samples of the windows are prepared by NTHREAD threads and the
//   frames (chunks of bins x windows) run asynchronously (batched by HailoRT)
#define NTHREAD 3
static int search_multi(const float **xs, const int8_t **x8s, int NW,
    const float *code, int M, double fs, const double *fc, int K, int nout,
    float *P, int ldP, sdr_acc_prof_t *prof)
{
    int fa[3][2] = {{0}}, A, B;
    if (!vdev || K < 1 || NW < 1) return 0;
    if (factorize8(M, &fa[0][0], &fa[0][1])) {
        fa[1][0] = fa[0][1]; fa[1][1] = fa[0][0]; // swapped (A > B)
    }
    factorize(M, &fa[2][0], &fa[2][1]);
    std::lock_guard<std::mutex> lock(call_mtx);
    double t0 = now();

    // uniform bins spaced by fs / M (tolerance for single precision Doppler)
    for (int k = 1; k < K; k++) {
        if (fabs((fc[k] - fc[0]) * M / fs - k) > 1e-3) return 0;
    }
    // model of the HEFs of M with the fewest chunks of K bins, then the shared
    // DFT models, then by the factorization (multiples of 8, swapped,
    // factorize(), others of the HEFs in hef_dir), then the smallest Kc and R,
    // R >= nout / B (R < nout / B +
    // 16, R padded for the output features to avoid the corrupted outputs of
    // some multi-context HEFs) (a frame takes about 0.8 ms or more by the
    // overhead of the device, nearly independent of Kc for small M, the bins
    // >= K are not used)
    hmodel_t *m = NULL;
    int mrank[5] = {0};
    for (auto &h : hefs) {
        if (h[0] != M) continue;
        hmodel_t *mi = get_model(M, h[1], h[2], h[3]);
        if (!mi) continue;
        int R = (nout + mi->B - 1) / mi->B, j = 0;
        if (mi->R < R || mi->R >= R + 16) continue;
        while (j < 3 && (mi->A != fa[j][0] || mi->B != fa[j][1])) j++;
        int rank[5] = {(K + mi->Kc - 1) / mi->Kc, !mi->sh, j, mi->Kc, mi->R};
        if (!m || std::lexicographical_compare(rank, rank + 5, mrank, mrank + 5)) {
            m = mi;
            std::copy(rank, rank + 5, mrank);
        }
    }
    if (!m) return 0;
    A = m->A; B = m->B;
    int Kc = m->Kc, nch = (K + Kc - 1) / Kc, F = NW * nch;

    // RMS of the windows and code DFT normalized by max magnitude
    std::vector<double> sx(NW, 0.0);
    double cmax = 0.0;
    for (int w = 0; w < NW; w++) {
        for (int i = 0; i < 2 * M; i++) {
            double v = xs ? xs[w][i] : x8s[w][i];
            sx[w] += v * v;
        }
        sx[w] = sqrt(sx[w] / M);
        if (sx[w] <= 0.0) return 0;
    }
    for (int i = 0; i < M; i++) {
        cmax = std::max(cmax, (double)hypot(code[2*i], code[2*i+1]));
    }
    if (cmax <= 0.0) return 0;
    auto quant = [](double v, float sc, float zp) -> uint8_t {
        double q = v / sc + zp + 0.5;
        return (uint8_t)(q < 0.0 ? 0.0 : (q > 255.0 ? 255.0 : q));
    };
    m->cr.resize(M); m->ci.resize(M);
    for (int k1 = 0; k1 < A; k1++) for (int k2 = 0; k2 < B; k2++) { // [k1][k2]
        int i = k2 * A + k1; // code c.reshape(B, A) [k2][k1] -> NHWC [k1][k2]
        m->cr[k1 * B + k2] = quant(code[2*i] / cmax, m->qs[2], m->qz[2]);
        m->ci[k1 * B + k2] = quant(code[2*i+1] / cmax, m->qs[3], m->qz[3]);
    }
    double tA = now();
    // IF samples of the windows wiped by the carrier of the first bin and the
    // code DFT of chunk c rolled by c * Kc bins (shared DFT identity:
    // |IDFT(X[m+s] C[m])| = |IDFT(X[m] C[m-s])|), so the host prepares the IF
    // samples once per window instead of once per frame. Frames f = c * NW + w
    // (chunk c, window w) run as jobs of one frame (batched by HailoRT, faster
    // than larger jobs), the code DFT of chunks in a ring of RING slots (a slot
    // reused after the jobs of its frames are done)
    const int RING = 32;
    int nslot = std::min(RING, nch);
    if ((int)m->xr.size() < NW) { m->xr.resize(NW); m->xi.resize(NW); }
    for (int w = 0; w < NW; w++) { m->xr[w].resize(M); m->xi[w].resize(M); }
    if ((int)m->crs.size() < nslot) { m->crs.resize(nslot); m->cis.resize(nslot); }
    for (int i = 0; i < nslot; i++) { m->crs[i].resize(M); m->cis[i].resize(M); }
    if ((int)m->out.size() < F) m->out.resize(F);
    for (int f = 0; f < F; f++) m->out[f].resize((size_t)B * Kc * m->R);
    std::vector<float> tab(2 * M);
    carr_table(fc[0] / fs, M, B, tab.data());
    if (m->sh) get_plan(M);
    std::vector<std::thread> th;
    for (int t = 0; t < NTHREAD; t++) {
        th.emplace_back([&, t]() {
            fftwf_complex *in = NULL, *out = NULL;
            if (m->sh) { in = fftwf_alloc_complex(M); out = fftwf_alloc_complex(M); }
            for (int w = t; w < NW; w += NTHREAD) {
                if (!m->sh) {
                    prep_frame(m, xs ? xs[w] : NULL, x8s ? x8s[w] : NULL, tab.data(),
                        sx[w], m->xr[w].data(), m->xi[w].data());
                }
                else if (xs) {
                    prep_frame_sh(m, xs[w], tab.data(), sx[w], in, out,
                        m->xr[w].data(), m->xi[w].data());
                }
                else {
                    prep_frame_sh(m, x8s[w], tab.data(), sx[w], in, out,
                        m->xr[w].data(), m->xi[w].data());
                }
            }
            if (in) { fftwf_free(in); fftwf_free(out); }
        });
    }
    for (auto &t : th) t.join();
    double tB = now(), t1 = tB, tp = 0.0;
    // P[k][B*p1+p2] += out[p2][k*R+p1] * 64 / M^2 * sx^2 * cmax^2 by chunk
    // (as the jobs of the frames of the chunk are done, overlapped with the
    // inference of the next chunks, the windows added to a row of P in the
    // cache, tiles of ATILE rows p1 transposed into a buffer, bins by threads)
    int Rm = m->R, W = Kc * Rm;
    std::vector<float> tbs((size_t)NTHREAD * ATILE * B);
    auto acc_bin = [&](int k, float *t) {
        int c = k / Kc, kk = k % Kc;
        float ozp = m->oz, *p = P + (size_t)k * ldP;
        for (int w = 0; w < NW; w++) {
            const uint16_t *ok = m->out[c * NW + w].data() + kk * Rm;
            float osc = (float)(64.0 / ((double)M * M) * sx[w] * sx[w] * cmax * cmax * m->os);
            for (int p1b = 0; p1b < Rm && B * p1b < nout; p1b += ATILE) {
                int nt = std::min(Rm, p1b + ATILE) - p1b;
                for (int p2 = 0; p2 < B; p2++) {
                    const uint16_t *q = ok + (size_t)p2 * W + p1b;
                    for (int i = 0; i < nt; i++) t[i * B + p2] = (float)q[i] - ozp;
                }
                for (int i = 0; i < nt; i++) {
                    int n0 = B * (p1b + i), ne = std::min(B, nout - n0);
                    float *__restrict pp = p + n0;
                    const float *__restrict ti = t + (size_t)i * B;
                    for (int j = 0; j < ne; j++) pp[j] += ti[j] * osc;
                }
            }
        }
    };
    auto acc_chunk = [&](int c) { // bins of the chunk by threads
        int k0 = c * Kc, nk = std::min(Kc, K - k0), nt = std::min(NTHREAD, nk);
        std::vector<std::thread> ta;
        for (int t = 1; t < nt; t++) {
            ta.emplace_back([&, t]() {
                for (int k = k0 + t; k < k0 + nk; k += nt) acc_bin(k, tbs.data() + (size_t)t * ATILE * B);
            });
        }
        for (int k = k0; k < k0 + nk; k += nt) acc_bin(k, tbs.data());
        for (auto &t : ta) t.join();
    };
    // done flags of the frames set by the callbacks or by wait(), tagged by the
    // generation of the call (gok: OK, gerr: error) against late callbacks
    if (m->ndone < F) {
        m->done.emplace_back(new std::atomic<uint32_t>[F]);
        for (int f = 0; f < F; f++) m->done.back()[f] = 0;
        m->ndone = F;
    }
    std::atomic<uint32_t> *done = m->done.back().get();
    uint32_t gok = (m->gen += 4) + 1, gerr = gok + 1;
    int fdone = 0, cacc = 0; // frames done in order, next chunk accumulated
    double tacc = 0.0;
    auto acc_done = [&](int nf) { // accumulate the chunks of done frames < nf
        double ta = now();
        while (fdone < nf && done[fdone] == gok) fdone++;
        for (; cacc < nch && (cacc + 1) * NW <= fdone; cacc++) acc_chunk(cacc);
        tacc += now() - ta;
        return fdone >= nf || done[fdone] != gerr;
    };
    std::vector<AsyncInferJob> jobs;
    std::vector<int> job_c;  // chunk of the frame of job
    size_t jdone = 0;
    int noinfer = getenv("SDR_ACC_HAILO_NOINFER") != NULL;
    for (int c = 0; c < nch; c++) {
        int slot = c % nslot;
        while (jdone < jobs.size() && job_c[jdone] <= c - nslot) { // free the slot
            if (jobs[jdone].wait(std::chrono::milliseconds(TIMEOUT_MS)) != HAILO_SUCCESS) {
                fprintf(stderr, "sdr_acc_hailo: inference error\n");
                return 0;
            }
            done[jdone++] = gok; // done by wait() (the callback may be later)
        }
        double ta = now();
        // code DFT rolled: C[(k1 + A k2 - s) mod M] = base[k1'][(k2 - r) mod B]
        int sft = (c * Kc) % M, s1 = sft % A, s2 = sft / A;
        uint8_t *cr = m->crs[slot].data(), *ci = m->cis[slot].data();
        for (int k1 = 0; k1 < A; k1++) { // [k1][k2] layout
            int k1s = k1 >= s1 ? k1 - s1 : k1 - s1 + A, r = (k1 >= s1 ? s2 : s2 + 1) % B;
            const uint8_t *br = m->cr.data() + (size_t)k1s * B, *bi = m->ci.data() + (size_t)k1s * B;
            uint8_t *dr = cr + (size_t)k1 * B, *di = ci + (size_t)k1 * B;
            memcpy(dr + r, br, B - r); memcpy(dr, br + B - r, r);
            memcpy(di + r, bi, B - r); memcpy(di, bi + B - r, r);
        }
        tp += now() - ta;
        if (noinfer) continue;
        for (int w = 0; w < NW; w++) {
            int f = c * NW + w;
            auto b = m->cim->create_bindings();
            if (!b) return 0;
            Bindings bd = b.release();
            bd.input(m->in_name[0])->set_buffer(MemoryView(m->xr[w].data(), M));
            bd.input(m->in_name[1])->set_buffer(MemoryView(m->xi[w].data(), M));
            bd.input(m->in_name[2])->set_buffer(MemoryView(cr, M));
            bd.input(m->in_name[3])->set_buffer(MemoryView(ci, M));
            bd.output(m->out_name)->set_buffer(MemoryView(m->out[f].data(), m->out[f].size() * 2));
            if (!acc_done(f)) {
                fprintf(stderr, "sdr_acc_hailo: inference error\n");
                return 0;
            }
            if (m->cim->wait_for_async_ready(std::chrono::milliseconds(TIMEOUT_MS), 1) != HAILO_SUCCESS) {
                return 0;
            }
            std::atomic<uint32_t> *dp = done + f;
            auto job = m->cim->run_async({bd}, [dp, gok, gerr](const AsyncInferCompletionInfo &info) {
                *dp = info.status == HAILO_SUCCESS ? gok : gerr;
            });
            if (!job) return 0;
            jobs.push_back(job.release());
            job_c.push_back(c);
        }
    }
    double t2 = now();
    for (size_t i = jdone; i < jobs.size(); i++) {
        if (jobs[i].wait(std::chrono::milliseconds(TIMEOUT_MS)) != HAILO_SUCCESS) {
            fprintf(stderr, "sdr_acc_hailo: inference error\n");
            return 0;
        }
        done[i] = gok;
        acc_done((int)i + 1);
    }
    if (!noinfer) for (; cacc < nch; cacc++) acc_chunk(cacc); // all jobs waited
    // code offset 0 (DC of the IDFTs) biased by the quantization of the model
    // (+40-250% of the mean power, independent of the input): replaced by the
    // neighbors (circular for nout = M / 2)
    for (int k = 0; k < K && nout > 1; k++) {
        float *p = P + (size_t)k * ldP;
        p[0] = nout == M / 2 ? 0.5f * (p[1] + p[nout - 1]) : p[1];
    }
    double t3 = now();
    if (getenv("SDR_ACC_HAILO_DEBUG")) {
        fprintf(stderr, "sdr_acc_hailo: F=%d prep %.2f acc %.2f (tail %.2f) ms\n", F,
            (tB - tA + tp) * 1e3, tacc * 1e3, (t3 - t2) * 1e3);
    }
    if (prof) {
        prof->t_in = t1 - t0 + tp;
        prof->t_exec = t2 - t1 - tp;
        prof->t_out = t3 - t2;
        prof->bytes_in = (double)F * 4 * M;
        prof->bytes_out = (double)F * B * W * 2;
    }
    return 1;
}

static int search_code(const float *x, const int8_t *x8, const float *code,
    int M, double fs, const double *fc, int K, int nout, float *P, int ldP,
    sdr_acc_prof_t *prof)
{
    return search_multi(x ? &x : NULL, x8 ? &x8 : NULL, 1, code, M, fs, fc, K,
        nout, P, ldP, prof);
}

// parallel code search of NW windows (extension, see search_multi()) ---------
extern "C" int sdr_acc_hailo_search_multi(const float **xs, const int8_t **x8s,
    int NW, const float *code, int M, double fs, const double *fc, int K,
    int nout, float *P, int ldP, sdr_acc_prof_t *prof)
{
    return search_multi(xs, x8s, NW, code, M, fs, fc, K, nout, P, ldP, prof);
}

static int acc_search_code(const float *x, const float *code, int M, double fs,
    const double *fc, int K, int nout, float *P, int ldP, sdr_acc_prof_t *prof)
{
    return search_code(x, NULL, code, M, fs, fc, K, nout, P, ldP, prof);
}

static int acc_search_code_i8(const int8_t *x, const float *code, int M,
    double fs, const double *fc, int K, int nout, float *P, int ldP,
    sdr_acc_prof_t *prof)
{
    return search_code(NULL, x, code, M, fs, fc, K, nout, P, ldP, prof);
}

static int acc_search_code_i8_multi(const int8_t **x, int NW, const float *code,
    int M, double fs, const double *fc, int K, int nout, float *P, int ldP,
    sdr_acc_prof_t *prof)
{
    return search_multi(NULL, x, NW, code, M, fs, fc, K, nout, P, ldP, prof);
}

// initialize / finalize -------------------------------------------------------
static int acc_init(const char *opt)
{
    hef_dir.clear();
    batch_size = 0;
    std::string s = opt ? opt : "";
    for (size_t p = 0; p < s.size(); ) {
        size_t q = s.find(',', p);
        std::string t = s.substr(p, q == std::string::npos ? q : q - p);
        if (t.compare(0, 4, "hef=") == 0) hef_dir = t.substr(4);
        else if (t.compare(0, 6, "batch=") == 0) batch_size = atoi(t.c_str() + 6);
        else if (!t.empty()) fprintf(stderr, "sdr_acc_hailo: unknown option %s\n",
            t.c_str());
        if (q == std::string::npos) break;
        p = q + 1;
    }
    const char *env;
    if (hef_dir.empty() && (env = getenv("SDR_ACC_HAILO_HEF"))) hef_dir = env;
    if (hef_dir.empty()) { // <plugin dir>/hailo
        Dl_info info;
        if (dladdr((void *)acc_init, &info) && info.dli_fname) {
            std::string f = info.dli_fname;
            size_t i = f.find_last_of('/');
            hef_dir = (i == std::string::npos ? std::string(".") : f.substr(0, i)) + "/hailo";
        }
    }
    hefs.clear();
    if (DIR *dp = opendir(hef_dir.c_str())) {
        while (struct dirent *e = readdir(dp)) {
            int M, Kc, R, n = 0, ns = 0;
            if (sscanf(e->d_name, "acq_M%d_K%d_R%d_h.hef%n", &M, &Kc, &R, &n) == 3 &&
                n == (int)strlen(e->d_name)) hefs.push_back({M, Kc, R, 0});
            else if (sscanf(e->d_name, "acq_M%d_K%d_R%d_hs.hef%n", &M, &Kc, &R, &ns) == 3 &&
                ns == (int)strlen(e->d_name)) hefs.push_back({M, Kc, R, 1});
        }
        closedir(dp);
        std::sort(hefs.begin(), hefs.end());
    }
    auto v = VDevice::create();
    if (!v) {
        fprintf(stderr, "sdr_acc_hailo: no Hailo device\n");
        return 0;
    }
    vdev = v.release();
    return 1;
}

static void acc_exit(void)
{
    for (auto &p : fplans) fftwf_destroy_plan(p.second);
    fplans.clear();
    hmodels.clear();
    vdev.reset();
}

static const sdr_acc_plugin_t plugin = {
    SDR_ACC_API_VER, "hailo", acc_init, acc_exit, acc_search_code, NULL,
    acc_search_code_i8, acc_search_code_i8_multi
};

extern "C" const sdr_acc_plugin_t *sdr_acc_plugin(void)
{
    return &plugin;
}
