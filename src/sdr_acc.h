//
//  Pocket SDR C Library - Acquisition Accelerator Plugin Interface.
//
//  An accelerator backend (NPU etc.) of the parallel code search is built as a
//  shared library libsdr_acc_<name>.so (.dll) exporting sdr_acc_plugin(). The
//  interface uses plain tensors only, so a backend does not depend on the
//  receiver data structures of Pocket SDR.
//
//  Author:
//  T.SUZUKI
//
//  History:
//  2026-10-16  1.0  new
//  2026-10-23  1.1  API version 2 (search_code_peak(), optional)
//
#ifndef SDR_ACC_H
#define SDR_ACC_H

#ifdef __cplusplus
extern "C" {
#endif

#define SDR_ACC_API_VER 2       // plugin API version (1: no search_code_peak())

typedef struct {                // accelerator profile of a call
    double t_in;                // input tensor preparation time (s)
    double t_exec;              // accelerator execution time (s)
    double t_out;               // output tensor post-processing time (s)
    double bytes_in;            // bytes of input tensors
    double bytes_out;           // bytes of output tensors
} sdr_acc_prof_t;

typedef struct {                // accelerator backend plugin
    int api_ver;                // plugin API version (SDR_ACC_API_VER)
    const char *name;           // backend name

    // initialize backend (opt: backend option string) (return 1: OK, 0: error)
    int (*init)(const char *opt);

    // finalize backend
    void (*exit)(void);

    // parallel code search for one non-coherent integration step
    //   x    (I) IF samples, complex int8 [M][2] (I,Q) (before carrier wipe-off)
    //   code (I) conjugate DFT of code replica, complex float [M][2]
    //   M    (I) number of samples (DFT size)
    //   fs   (I) sampling frequency (Hz)
    //   fc   (I) carrier frequencies of Doppler bins (Hz) [K] (phase 0 at x[0])
    //   K    (I) number of Doppler bins
    //   nout (I) number of code phases to output (<= M)
    //   P    (IO) correlation powers [K][ldP], P[k*ldP+n] += |corr_k(n)|^2
    //            corr_k = IDFT(DFT(x * exp(-j*2*pi*fc[k]*t)) * code) / M^2
    //            (DFT/IDFT without normalization, same as sdr_search_code())
    //   ldP  (I) leading dimension of P
    //   prof (O) profile of the call (NULL: not output)
    // (return 1: OK, 0: unsupported shape or error -> CPU fallback)
    int (*search_code)(const int8_t *x, const float *code, int M, double fs,
        const double *fc, int K, int nout, float *P, int ldP,
        sdr_acc_prof_t *prof);

    // parallel code search of NW windows x [NW][M][2] with non-coherent
    // integration of all windows (other arguments are same as search_code())
    // (NULL: not supported, search_code() used for each window)
    int (*search_code_multi)(const int8_t **x, int NW, const float *code,
        int M, double fs, const double *fc, int K, int nout, float *P,
        int ldP, sdr_acc_prof_t *prof);

    // parallel code search of NW windows as search_code_multi() reduced to the
    // peak of the correlation powers P[K][nout] (P not output)
    //   Pk   (O) Pk[0]: max of P, Pk[1]: mean of P, Pk[2], Pk[3]: P at the
    //            code offset of the max in the bins ix[0]-1, ix[0]+1 (0: out
    //            of the bins)
    //   ix   (O) ix[0]: bin, ix[1]: code offset of the max
    // (NULL: not supported, API version 2)
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
