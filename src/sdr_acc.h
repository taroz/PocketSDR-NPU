//
//  Pocket SDR C Library - Accelerator Backend Plugin Interface.
//
//  A hardware-specific accelerator backend (NPU etc.) is built as a separate
//  shared library (libsdr_acc_<name>.so/.dll) exporting sdr_acc_plugin().
//  The interface uses plain tensors only, so a backend does not depend on the
//  receiver data structures of Pocket SDR (see doc/npu_architecture.md).
//
//  Author:
//  T.SUZUKI
//
//  History:
//  2026-10-02  1.0  new
//  2026-10-03  1.1  add batched tracking correlator (API version 2)
//  2026-10-03  1.2  add parallel code search of int8 IF samples (API version 3)
//  2026-10-11  1.3  add parallel code search of multiple windows (API version 4)
//  2026-10-27  1.4  add parallel code search reduced to the peak (API version 5)
//
#ifndef SDR_ACC_H
#define SDR_ACC_H

#ifdef __cplusplus
extern "C" {
#endif

#define SDR_ACC_API_VER 5       // plugin API version
#define SDR_ACC_API_MIN 3       // min plugin API version supported

typedef struct {                // accelerator profile of a call
    double t_in;                // input tensor preparation time (s)
    double t_exec;              // accelerator execution time (s)
    double t_out;               // output tensor post-processing time (s)
    double bytes_in;            // bytes of input tensors
    double bytes_out;           // bytes of output tensors
} sdr_acc_prof_t;

typedef struct {                // tracking correlator of a channel
    double fc;                  // IF carrier frequency (Hz)
    double phi;                 // IF carrier phase at x[0] (cyc)
    int n;                      // number of correlators
    int scale;                  // code amplitude scale (max |code| <= 127)
    const int8_t **code;        // resampled code of correlators [n][N]
    const int *j;               // code wrap-around sample of correlators [n]
    float *corr;                // correlations [n][2][2] (output)
                                // corr[i][0] (before wrap-around, s < j[i]):
                                //   sum(x[s]*carr[s]*code[i][s+N-j[i]])
                                // corr[i][1] (after wrap-around, s >= j[i]):
                                //   sum(x[s]*carr[s]*code[i][s-j[i]])
                                // carr[s] = exp(-j*2*pi*(phi+fc*s/fs))
} sdr_acc_corr_t;

typedef struct {                // accelerator backend plugin
    int api_ver;                // plugin API version (SDR_ACC_API_VER)
    const char *name;           // backend name

    // initialize backend (opt: backend option string, e.g. device name)
    // (return 1: OK, 0: error) (may be called again for acquisition and
    // tracking backends, exit() is called for each init())
    int (*init)(const char *opt);

    // finalize backend
    void (*exit)(void);

    // parallel code search for one non-coherent integration step
    //   x    (I) IF samples, complex float [M][2] (before carrier wipe-off)
    //   code (I) conjugate DFT of code replica, complex float [M][2]
    //   M    (I) number of samples (DFT size)
    //   fs   (I) sampling frequency (Hz)
    //   fc   (I) carrier frequencies of Doppler bins (Hz) [K] (phase 0 at x[0])
    //   K    (I) number of Doppler bins
    //   nout (I) number of code phases to output (<= M)
    //   P    (IO) correlation powers [K][ldP], P[k*ldP+n] += |corr_k(n)|^2
    //            corr_k = IDFT(DFT(x * exp(-j*2*pi*fc[k]*t)) * code) / M^2
    //            (DFT/IDFT without normalization, same as the CPU reference)
    //   ldP  (I) leading dimension of P
    //   prof (O) profile of the call (NULL: not output)
    // (return 1: OK, 0: unsupported shape or error -> CPU fallback)
    int (*search_code)(const float *x, const float *code, int M, double fs,
        const double *fc, int K, int nout, float *P, int ldP,
        sdr_acc_prof_t *prof);

    // batched tracking correlators of channels sharing the same IF samples
    //   x    (I) IF samples, complex int8 [N][2] (I,Q)
    //   N    (I) number of samples (correlation length)
    //   fs   (I) sampling frequency (Hz)
    //   nch  (I) number of channels
    //   corr (IO) correlators of channels [nch] (see sdr_acc_corr_t)
    //   prof (O) profile of the call (NULL: not output)
    // (return 1: OK, 0: unsupported or error -> CPU fallback)
    // (NULL: not supported by backend)
    int (*corr_batch)(const int8_t *x, int N, double fs, int nch,
        sdr_acc_corr_t *corr, sdr_acc_prof_t *prof);

    // parallel code search of int8 IF samples x [M][2] (I,Q) (other arguments
    // are same as search_code()) (NULL: not supported, search_code() used)
    int (*search_code_i8)(const int8_t *x, const float *code, int M,
        double fs, const double *fc, int K, int nout, float *P, int ldP,
        sdr_acc_prof_t *prof);

    // parallel code search of NW windows of int8 IF samples x [NW][M][2]
    // (I,Q) with non-coherent integration of all windows (other arguments
    // are same as search_code()) (NULL: not supported, search_code_i8() or
    // search_code() used for each window) (API version >= 4)
    int (*search_code_i8_multi)(const int8_t **x, int NW, const float *code,
        int M, double fs, const double *fc, int K, int nout, float *P,
        int ldP, sdr_acc_prof_t *prof);

    // parallel code search of NW windows as search_code_multi() reduced to the
    // peak of the correlation powers P[K][nout] (P not output)
    //   Pk   (O) Pk[0]: max of P, Pk[1]: mean of P, Pk[2], Pk[3]: P at the
    //            code offset of the max in the bins ix[0]-1, ix[0]+1 (0: out
    //            of the bins)
    //   ix   (O) ix[0]: bin, ix[1]: code offset of the max
    // (NULL: not supported) (API version >= 5)
    int (*search_code_peak)(const int8_t **x, int NW, const float *code, int M,
        double fs, const double *fc, int K, int nout, float *Pk, int *ix,
        sdr_acc_prof_t *prof);
} sdr_acc_plugin_t;

// plugin entry point (exported by backend shared library)
typedef const sdr_acc_plugin_t *(*sdr_acc_plugin_func_t)(void);

#ifdef __cplusplus
}
#endif
#endif // SDR_ACC_H
