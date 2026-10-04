// FIR decimator (channelizer) on CPU (AVX2) and NPU (OpenVINO strided conv)
//
// usage: fir_dec file rfch toff tlen D ntap fc_MHz blk_ms [out_int8x2]
//   input: Pocket SDR FE 8CH RAW32 (2 bit IQ), fs = 16 MHz
// build: g++ -O3 -std=c++17 -mavx2 -mfma -I$OV_DIR/runtime/include fir_dec.cpp
//        -o fir_dec -L$OV_DIR/runtime/lib/intel64 -lopenvino
#include <openvino/openvino.hpp>
#include <openvino/opsets/opset13.hpp>
#include <openvino/core/preprocess/pre_post_process.hpp>
#include <immintrin.h>
#include <sys/resource.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
namespace o = ov::opset13;

static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
static double pcpu() { rusage r; getrusage(RUSAGE_SELF, &r); return r.ru_utime.tv_sec + r.ru_utime.tv_usec * 1e-6 + r.ru_stime.tv_sec + r.ru_stime.tv_usec * 1e-6; }

// read RF channel of RAW32 as int8 I/Q planar
static int read_if(const char *file, int rfch, double fs, double toff, double tlen,
    std::vector<int8_t> &I, std::vector<int8_t> &Q)
{
    static const int8_t valI[] = {1, 3, -1, -3}, valQ[] = {-1, -3, 1, 3};
    long N = (long)(fs * tlen);
    FILE *fp = fopen(file, "rb");
    if (!fp) return 0;
    std::vector<uint8_t> raw(4 * N);
    fseeko(fp, (off_t)(fs * toff) * 4, SEEK_SET);
    if (fread(raw.data(), 4, N, fp) < (size_t)N) { fclose(fp); return 0; }
    fclose(fp);
    I.resize(N); Q.resize(N);
    for (long i = 0; i < N; i++) {
        uint8_t b = raw[i * 4 + rfch / 2];
        I[i] = valI[(b >> (rfch % 2 ? 4 : 0)) & 3];
        Q[i] = valQ[(b >> (rfch % 2 ? 6 : 2)) & 3];
    }
    return 1;
}

// lowpass FIR (windowed sinc, Hamming), unit DC gain
static std::vector<float> design(int ntap, double fc, double fs)
{
    std::vector<float> h(ntap);
    double sum = 0.0, fn = fc / fs;
    for (int i = 0; i < ntap; i++) {
        double t = i - (ntap - 1) / 2.0;
        double s = t == 0.0 ? 2.0 * fn : sin(2.0 * M_PI * fn * t) / (M_PI * t);
        h[i] = (float)(s * (0.54 - 0.46 * cos(2.0 * M_PI * i / (ntap - 1))));
        sum += h[i];
    }
    for (auto &v : h) v = (float)(v / sum);
    return h;
}

// CPU decimator (AVX2): y[n] = sum_k h[k] x[n*D+k] (x: float planar)
static void fir_cpu(const float *x, long nout, int D, const std::vector<float> &h, float *y)
{
    int nt = (int)h.size();
    for (long n = 0; n < nout; n++) {
        const float *xp = x + n * D;
        __m256 acc = _mm256_setzero_ps();
        int k = 0;
        for ( ; k + 8 <= nt; k += 8) {
            acc = _mm256_fmadd_ps(_mm256_loadu_ps(h.data() + k), _mm256_loadu_ps(xp + k), acc);
        }
        __m128 s = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
        s = _mm_hadd_ps(s, s); s = _mm_hadd_ps(s, s);
        float v = _mm_cvtss_f32(s);
        for ( ; k < nt; k++) v += h[k] * xp[k];
        y[n] = v;
    }
}

int main(int argc, char **argv)
{
    const char *file = argv[1];
    int rfch = atoi(argv[2]) - 1;
    double toff = atof(argv[3]), tlen = atof(argv[4]), fs = 16e6;
    int D = atoi(argv[5]), ntap = atoi(argv[6]);
    double fc = atof(argv[7]) * 1e6, blk_ms = atof(argv[8]);
    const char *out = argc > 9 ? argv[9] : "";
    std::vector<int8_t> I, Q;
    if (!read_if(file, rfch, fs, toff, tlen, I, Q)) { fprintf(stderr, "read error\n"); return 1; }
    long N = (long)I.size(), nout = (N - ntap) / D + 1;
    auto h = design(ntap, fc, fs);
    printf("input %ld samples (%.1f s), D=%d (%.2f MHz), %d taps, fc=%.2f MHz\n", N, tlen, D, fs / D * 1e-6, ntap, fc * 1e-6);

    // CPU: int8 -> float (block-wise) + FIR (as a receiver would process blocks)
    std::vector<float> yI(nout), yQ(nout);
    {
        long blk = (long)(fs * blk_ms * 1e-3) / D * D, no = blk / D;
        std::vector<float> xf(blk + ntap);
        double t0 = now(), c0 = pcpu();
        for (long o0 = 0; o0 < nout; o0 += no) {
            long n = std::min(no, nout - o0), i0 = o0 * D, ni = (n - 1) * D + ntap;
            for (int c = 0; c < 2; c++) {
                const int8_t *src = (c ? Q.data() : I.data()) + i0;
                for (long i = 0; i < ni; i++) xf[i] = src[i];
                fir_cpu(xf.data(), n, D, h, (c ? yQ.data() : yI.data()) + o0);
            }
        }
        double tw = now() - t0, tc = pcpu() - c0;
        printf("CPU: time=%.3f s cpu=%.3f s -> core load %.1f%% (%.1f Msps in)\n", tw, tc, tc / tlen * 100.0, N / tw * 1e-6);
    }
    // NPU: strided GroupConvolution, int8 input [1,2,blk+ntap-1] -> [1,2,blk/D]
    std::vector<float> zI(nout, 0.f), zQ(nout, 0.f);
    {
        long blk = (long)(fs * blk_ms * 1e-3) / D * D, no = blk / D, ni = (no - 1) * D + ntap;
        auto x = std::make_shared<o::Parameter>(ov::element::f32, ov::Shape{1, 2, (size_t)ni});
        std::vector<float> hr(h.rbegin(), h.rend()); // conv = correlation in OpenVINO? (no flip)
        auto W = o::Constant::create(ov::element::f32, ov::Shape{2, 1, 1, (size_t)ntap}, std::vector<float>(2 * ntap));
        std::vector<float> w2(2 * ntap);
        for (int c = 0; c < 2; c++) for (int k = 0; k < ntap; k++) w2[c * ntap + k] = h[k];
        W = o::Constant::create(ov::element::f32, ov::Shape{2, 1, 1, (size_t)ntap}, w2);
        auto y = std::make_shared<o::GroupConvolution>(x, W, ov::Strides{(size_t)D},
            ov::CoordinateDiff{0}, ov::CoordinateDiff{0}, ov::Strides{1});
        auto m = std::make_shared<ov::Model>(ov::OutputVector{y}, ov::ParameterVector{x});
        ov::preprocess::PrePostProcessor ppp(m);
        ppp.input(0).tensor().set_element_type(ov::element::i8);
        ppp.output(0).tensor().set_element_type(ov::element::f16);
        ov::Core core;
        double t0 = now();
        auto cm = core.compile_model(ppp.build(), "NPU", ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY));
        auto rq = cm.create_infer_request();
        printf("NPU model compiled %.2f s (block %.0f ms, in %ld, out %ld)\n", now() - t0, blk_ms, ni, no);
        rq.infer();
        t0 = now(); double c0 = pcpu(), texec = 0.0;
        int8_t *tx = rq.get_input_tensor(0).data<int8_t>();
        for (long o0 = 0; o0 < nout; o0 += no) {
            long n = std::min(no, nout - o0), i0 = o0 * D, nin = std::min(ni, N - i0);
            memcpy(tx, I.data() + i0, nin);
            memcpy(tx + ni, Q.data() + i0, nin);
            if (nin < ni) { memset(tx + nin, 0, ni - nin); memset(tx + ni + nin, 0, ni - nin); }
            double e0 = now();
            rq.infer();
            texec += now() - e0;
            const ov::float16 *yo = rq.get_output_tensor(0).data<ov::float16>();
            for (long i = 0; i < n; i++) { zI[o0 + i] = yo[i]; zQ[o0 + i] = yo[no + i]; }
        }
        double tw = now() - t0, tc = pcpu() - c0;
        printf("NPU: time=%.3f s (exec %.3f s) cpu=%.3f s -> core load %.1f%% (%.1f Msps in)\n", tw, texec, tc, tc / tlen * 100.0, N / tw * 1e-6);
    }
    // compare
    double e = 0, s = 0;
    for (long i = 0; i < nout; i++) {
        e += (zI[i] - yI[i]) * (zI[i] - yI[i]) + (zQ[i] - yQ[i]) * (zQ[i] - yQ[i]);
        s += yI[i] * yI[i] + yQ[i] * yQ[i];
    }
    printf("NPU vs CPU: NRMSE %.5f (output RMS %.3f)\n", sqrt(e / s), sqrt(s / nout / 2));
    // write INT8X2 (CPU output, 4 bit range): bench reads (I, -Q)
    if (*out) {
        double rms = sqrt(s / nout / 2), sc = 2.5 / rms;
        FILE *fp = fopen(out, "wb");
        long nclip = 0;
        for (long i = 0; i < nout; i++) {
            int vi = (int)lrint(yI[i] * sc), vq = (int)lrint(yQ[i] * sc);
            if (abs(vi) > 7 || abs(vq) > 7) nclip++;
            int8_t b[2] = {(int8_t)std::max(-7, std::min(7, vi)), (int8_t)-std::max(-7, std::min(7, vq))};
            fwrite(b, 1, 2, fp);
        }
        fclose(fp);
        printf("written %s (%ld samples, %.2f MHz, clip %.4f%%)\n", out, nout, fs / D * 1e-6, nclip * 100.0 / nout);
    }
    return 0;
}
