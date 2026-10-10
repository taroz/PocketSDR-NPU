//
//  Pocket SDR - Accelerator Backend Plugin for Hailo NPU (HailoRT).
//
//  The parallel code search of the Doppler bins is executed on a Hailo NPU
//  (e.g. Hailo-10H of Raspberry Pi AI HAT+ 2) by a HEF compiled with the Hailo
//  Dataflow Compiler from the ONNX model of a carrier folded four-step DFT
//  (bins in the channels, batch 1, see doc/npu_architecture.md):
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
//  The INT8 model with the 16-bit power tail has the power NRMSE about 0.07
//  (1 ms codes of 10.23 Mcps) to 0.13 (L1CA) and the C/N0 within 0.2 dB of the
//  CPU reference.
//
//  options: [hef=dir][,batch=n]
//           batch=n: batch size of the models (default: HailoRT default)
//           hef=dir: directory of HEFs (default: $SDR_ACC_HAILO_HEF or
//                  <plugin dir>/hailo)
//  environment: SDR_ACC_HAILO_HEF: directory of HEFs
//
//  Author:
//  T.SUZUKI
//
//  History:
//  2026-10-11  1.0  new
//
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <dlfcn.h>
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <mutex>
#include <chrono>
#include <algorithm>
#include <thread>
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
    int M = 0, Kc = 0, R = 0, A = 0, B = 0;
    float qs[4], qz[4], os = 1.0f, oz = 0.0f; // quantization of inputs/output
    std::vector<std::vector<uint8_t>> xr, xi;  // per chunk
    std::vector<uint8_t> cr, ci;
    std::vector<std::vector<uint16_t>> out;    // per chunk
};
static int batch_size = 0;              // batch size (0: HailoRT default)

static std::unique_ptr<VDevice> vdev;
static std::map<std::string, std::unique_ptr<hmodel_t>> hmodels;
static std::string hef_dir;
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

// load HEF model for (M, Kc, R) -----------------------------------------------
static hmodel_t *get_model(int M, int Kc, int R)
{
    char key[64];
    snprintf(key, sizeof(key), "acq_M%d_K%d_R%d_h", M, Kc, R);
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
    m->M = M; m->Kc = Kc; m->R = R;
    factorize(M, &m->A, &m->B);
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
template <typename T>
static void prep_frame_t(int A, int B, const T *__restrict x,
    const float *__restrict tab, float sr, float si, float zr, float zi,
    uint8_t *__restrict xr, uint8_t *__restrict xi)
{
    for (int n1 = 0; n1 < A; n1++) {
        const float *t = tab + 2 * B * n1;
        const T *xp = x + 2 * B * n1;
        for (int n2 = 0; n2 < B; n2++) {
            float re = xp[2*n2], im = xp[2*n2+1], cr = t[2*n2], ci = t[2*n2+1];
            float q1 = (re * cr - im * ci) * sr + zr, q2 = (re * ci + im * cr) * si + zi;
            xr[n2 * A + n1] = (uint8_t)(q1 < 0.f ? 0.f : (q1 > 255.f ? 255.f : q1));
            xi[n2 * A + n1] = (uint8_t)(q2 < 0.f ? 0.f : (q2 > 255.f ? 255.f : q2));
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

// parallel code search of NW windows (non-coherent integration) -------------
//   the frames (windows x chunks of bins) are prepared by NTHREAD threads and
//   run asynchronously in groups of the async queue size (batched by HailoRT)
#define NTHREAD 3
static int search_multi(const float **xs, const int8_t **x8s, int NW,
    const float *code, int M, double fs, const double *fc, int K, int nout,
    float *P, int ldP, sdr_acc_prof_t *prof)
{
    int A, B;
    if (!vdev || K < 1 || NW < 1 || !factorize(M, &A, &B)) return 0;
    std::lock_guard<std::mutex> lock(call_mtx);
    double t0 = now();

    // uniform bins spaced by fs / M (tolerance for single precision Doppler)
    for (int k = 1; k < K; k++) {
        if (fabs((fc[k] - fc[0]) * M / fs - k) > 1e-3) return 0;
    }
    // model with the largest Kc <= K (7, 3, 1) and R = nout / B
    int R = (nout + B - 1) / B;
    hmodel_t *m = NULL;
    static const int kcs[] = {7, 3, 1};
    for (int i = 0; i < 3 && !m; i++) {
        if (kcs[i] <= K) m = get_model(M, kcs[i], R);
    }
    if (!m) return 0;
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
    // carrier tables of the chunks (phase 0 at x[0] of each window)
    std::vector<std::vector<float>> tabs(nch, std::vector<float>(2 * M));
    std::vector<std::thread> th;
    for (int c = 0; c < nch; c++) {
        th.emplace_back([&, c]() { carr_table(fc[c * Kc] / fs, M, B, tabs[c].data()); });
    }
    for (auto &t : th) t.join();
    // frames f = w * nch + c (window w, chunk c)
    if ((int)m->xr.size() < F) { m->xr.resize(F); m->xi.resize(F); m->out.resize(F); }
    auto prep = [&](int f0, int f1) { // prepare frames f0 to f1-1 by threads
        std::vector<std::thread> tp;
        for (int t = 0; t < NTHREAD; t++) {
            tp.emplace_back([&, t]() {
                for (int f = f0 + t; f < f1; f += NTHREAD) {
                    int w = f / nch, c = f % nch;
                    m->xr[f].resize(M); m->xi[f].resize(M);
                    m->out[f].resize((size_t)B * Kc * m->R);
                    prep_frame(m, xs ? xs[w] : NULL, x8s ? x8s[w] : NULL,
                        tabs[c].data(), sx[w], m->xr[f].data(), m->xi[f].data());
                }
            });
        }
        for (auto &t : tp) t.join();
    };
    double tB = now();
    for (int f = 0; f < F; f++) {
        m->xr[f].resize(M); m->xi[f].resize(M); m->out[f].resize((size_t)B * Kc * m->R);
    }
    std::vector<Bindings> binds;
    for (int f = 0; f < F; f++) {
        auto b = m->cim->create_bindings();
        if (!b) return 0;
        Bindings bd = b.release();
        bd.input(m->in_name[0])->set_buffer(MemoryView(m->xr[f].data(), M));
        bd.input(m->in_name[1])->set_buffer(MemoryView(m->xi[f].data(), M));
        bd.input(m->in_name[2])->set_buffer(MemoryView(m->cr.data(), M));
        bd.input(m->in_name[3])->set_buffer(MemoryView(m->ci.data(), M));
        bd.output(m->out_name)->set_buffer(MemoryView(m->out[f].data(), m->out[f].size() * 2));
        binds.push_back(bd);
    }
    double t1 = now(), tp = 0.0;
    if (getenv("SDR_ACC_HAILO_DEBUG")) fprintf(stderr, "sdr_acc_hailo: rms/code %.2f tab %.2f bind %.2f ms\n", (tA - t0) * 1e3, (tB - tA) * 1e3, (t1 - tB) * 1e3);
    auto qs = m->cim->get_async_queue_size();
    int qn = qs ? (int)qs.value() : 1;
    std::vector<AsyncInferJob> jobs;
    const int PG = NTHREAD * 4; // frames per preparation group
    for (int g0 = 0; g0 < F; g0 += PG) { // prepare and run groups (overlapped with the NPU)
        int g1 = std::min(F, g0 + PG);
        double ta = now();
        prep(g0, g1);
        tp += now() - ta;
        if (getenv("SDR_ACC_HAILO_DEBUG")) fprintf(stderr, "sdr_acc_hailo: prep %d-%d %.2f ms\n", g0, g1, (now() - ta) * 1e3);
        if (getenv("SDR_ACC_HAILO_NOINFER")) continue;
        for (int f = g0; f < g1; ) {
            int n = std::min(qn, g1 - f);
            if (m->cim->wait_for_async_ready(std::chrono::milliseconds(TIMEOUT_MS), n) != HAILO_SUCCESS) {
                return 0;
            }
            std::vector<Bindings> g(binds.begin() + f, binds.begin() + f + n);
            auto job = m->cim->run_async(g);
            if (!job) return 0;
            jobs.push_back(job.release());
            f += n;
        }
    }
    if (getenv("SDR_ACC_HAILO_DEBUG")) fprintf(stderr, "sdr_acc_hailo: F=%d queue=%d\n", F, qn);
    for (auto &j : jobs) {
        if (j.wait(std::chrono::milliseconds(TIMEOUT_MS)) != HAILO_SUCCESS) {
            fprintf(stderr, "sdr_acc_hailo: inference error\n");
            return 0;
        }
    }
    double t2 = now();

    // P[k][B*p1+p2] += out[p2][k*R+p1] * 64 / M^2 * sx^2 * cmax^2 (bins by threads)
    int Rm = m->R, W = Kc * Rm;
    th.clear();
    for (int t = 0; t < NTHREAD; t++) {
        th.emplace_back([&, t]() {
            for (int k = t; k < K; k += NTHREAD) {
                int c = k / Kc, kk = k % Kc;
                float *p = P + (size_t)k * ldP;
                for (int w = 0; w < NW; w++) {
                    const uint16_t *o = m->out[w * nch + c].data();
                    float osc = (float)(64.0 / ((double)M * M) * sx[w] * sx[w] * cmax * cmax * m->os);
                    float ozp = m->oz;
                    for (int p1 = 0; p1 < Rm; p1++) for (int p2 = 0; p2 < B; p2++) {
                        int n = B * p1 + p2;
                        if (n < nout) p[n] += ((float)o[(size_t)p2 * W + kk * Rm + p1] - ozp) * osc;
                    }
                }
            }
        });
    }
    for (auto &t : th) t.join();
    double t3 = now();
    if (prof) {
        prof->t_in = t1 - t0 + tp;
        prof->t_exec = t2 - t1 - tp;
        prof->t_out = t3 - t2;
        prof->bytes_in = (double)F * 2 * M + 2 * M;
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
    hmodels.clear();
    vdev.reset();
}

static const sdr_acc_plugin_t plugin = {
    SDR_ACC_API_VER, "hailo", acc_init, acc_exit, acc_search_code, NULL,
    acc_search_code_i8
};

extern "C" const sdr_acc_plugin_t *sdr_acc_plugin(void)
{
    return &plugin;
}
