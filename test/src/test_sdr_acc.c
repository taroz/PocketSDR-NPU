//
//  Unit tests for sdr_acc.c (acquisition accelerator backends).
//
//  The plugins in SDR_ACC_TEST (e.g. "rknn" or "hailo:hef=<dir>,batch=16",
//  separated by ';', default: none) are compared with the CPU (SDR_ACC_PATH:
//  directory of the plugins).
//
#include "test_sdr.h"

#define FS       8e6        // sampling frequency (Hz)
#define PRN      5          // PRN number
#define COFF     1234       // code offset (samples)
#define DOP      1000.0     // Doppler frequency (Hz)
#define AMP      0.25       // signal amplitude (noise sigma = 1)
#define NCALL    10         // number of non-coherent integrations
#define TOL_NRMSE 0.2       // tolerance of normalized RMS error of powers (INT8)
#define TOL_CN0  0.3        // tolerance of C/N0 metric (dB)

typedef struct {            // test data
    int N, M, len_fds;
    double T;
    sdr_buff_t *buff;
    sdr_cpx_t *code_fft;
    float *fds;
} test_data_t;

// normal random number (Box-Muller) -------------------------------------------
static double randn(void)
{
    double u1 = (rand() + 1.0) / (RAND_MAX + 2.0), u2 = rand() / (RAND_MAX + 1.0);
    return sqrt(-2.0 * log(u1)) * cos(2.0 * PI * u2);
}

// quantize to 4-bit IF sample -------------------------------------------------
static int8_t quant(double x)
{
    int v = (int)floor(x * 2.0 + 0.5);
    return (int8_t)(v < -7 ? -7 : (v > 7 ? 7 : v));
}

// generate test data (L1CA signal + noise) ------------------------------------
static void gen_data(test_data_t *d)
{
    int len_code;
    int8_t *code = sdr_gen_code("L1CA", PRN, &len_code);
    d->T = sdr_code_cyc("L1CA");
    d->N = (int)(FS * d->T);
    d->M = 2 * d->N;
    d->buff = sdr_buff_new(d->N * (NCALL + 1), 2);
    srand(1);
    for (int i = 0; i < d->buff->N; i++) {
        double t = (double)(i - COFF) / FS;
        int j = (int)floor(t / d->T * len_code) % len_code;
        if (j < 0) j += len_code;
        double ph = 2.0 * PI * DOP * i / FS;
        double I = AMP * code[j] * cos(ph) + randn();
        double Q = AMP * code[j] * sin(ph) + randn();
        d->buff->data[i] = SDR_CPX8(quant(I), quant(Q));
    }
    d->code_fft = sdr_cpx_malloc(d->M);
    sdr_gen_code_fft(code, NULL, len_code, d->T, 0.0, FS, d->N, d->N,
        d->code_fft);
    d->fds = sdr_dop_bins(d->T, 0.0f, 5000.0f, &d->len_fds);
}

// free test data --------------------------------------------------------------
static void free_data(test_data_t *d)
{
    sdr_buff_free(d->buff);
    sdr_cpx_free(d->code_fft);
    sdr_free(d->fds);
}

// non-coherent integrated correlation powers (multi: multi-window search) -----
static float *search(const test_data_t *d, int multi)
{
    float *P = (float *)calloc((size_t)d->M * d->len_fds, sizeof(float));
    int ix[NCALL];
    for (int i = 0; i < NCALL; i++) ix[i] = i * d->N;
    if (multi && sdr_acc_search_code_multi(d->code_fft, d->T, d->buff, ix,
        NCALL, d->M, d->N, FS, 0.0, d->fds, d->len_fds, P)) {
        return P;
    }
    for (int i = 0; i < NCALL; i++) {
        sdr_search_code2(d->code_fft, d->T, d->buff, ix[i], d->M, d->N, FS,
            0.0, d->fds, d->len_fds, P);
    }
    return P;
}

// normalized RMS error of powers (code phases < N) ----------------------------
static double nrmse(const test_data_t *d, const float *P0, const float *P1)
{
    double e = 0.0, s = 0.0;
    for (int i = 0; i < d->len_fds; i++) {
        for (int j = 0; j < d->N; j++) {
            double p0 = P0[i*d->M+j], p1 = P1[i*d->M+j];
            e += (p1 - p0) * (p1 - p0);
            s += p0 * p0;
        }
    }
    return sqrt(e / s);
}

// compare powers with CPU -----------------------------------------------------
static void check_powers(const test_data_t *d, const float *P_ref,
    const float *P, const char *name)
{
    int ix0[2], ix1[2];
    float cn0_ref = sdr_corr_max(P_ref, d->M, d->N, d->len_fds, d->T, ix0);
    float cn0 = sdr_corr_max(P, d->M, d->N, d->len_fds, d->T, ix1);
    double e = nrmse(d, P_ref, P);
    printf("[%s nrmse=%.4f cn0=%.2f/%.2f] ", name, e, cn0_ref, cn0);
    TEST_ASSERT_EQ_INT(ix0[0], ix1[0]);
    TEST_ASSERT_EQ_INT(ix0[1], ix1[1]);
    TEST_ASSERT_TRUE(e < TOL_NRMSE);
    TEST_ASSERT_NEAR(cn0_ref, cn0, TOL_CN0);
}

// CPU backend and plugin load error -------------------------------------------
static void test_sdr_acc_cpu(void)
{
    test_data_t d;
    int ix[2];
    gen_data(&d);
    TEST_ASSERT_EQ_INT(1, sdr_acc_init("cpu"));
    TEST_ASSERT_EQ_INT(SDR_ACC_CPU, sdr_acc_type(NULL));
    TEST_ASSERT_EQ_INT(0, sdr_acc_search_multi());
    float *P = search(&d, 1);
    sdr_corr_max(P, d.M, d.N, d.len_fds, d.T, ix);
    TEST_ASSERT_EQ_INT(COFF, ix[1]);
    TEST_ASSERT_EQ_INT(0, sdr_acc_init("no_such_plugin"));
    TEST_ASSERT_EQ_INT(SDR_ACC_CPU, sdr_acc_type(NULL));
    free(P);
    free_data(&d);
}

// plugin backends vs CPU (per window and multi-window) ------------------------
static void test_sdr_acc_plugin(void)
{
    const char *env = getenv("SDR_ACC_TEST");
    char list[1024], *p, *q;
    test_data_t d;
    if (!env || !*env) {
        printf("[skipped: SDR_ACC_TEST not set] ");
        return;
    }
    gen_data(&d);
    sdr_acc_init("cpu");
    float *P_ref = search(&d, 0);
    snprintf(list, sizeof(list), "%s", env);
    for (p = list; p && *p; p = q) {
        if ((q = strchr(p, ';'))) *q++ = '\0';
        TEST_ASSERT_EQ_INT(1, sdr_acc_init(p));
        float *P1 = search(&d, 0), *P2 = search(&d, 1);
        sdr_acc_stat_t st;
        sdr_acc_stat(&st, 1);
        check_powers(&d, P_ref, P1, p);
        check_powers(&d, P_ref, P2, sdr_acc_search_multi() ? "multi" : "-");
        TEST_ASSERT_TRUE(st.ncall > 0 && st.nfall == 0);
        free(P1);
        free(P2);
        sdr_acc_exit();
    }
    free(P_ref);
    free_data(&d);
}

int main(void)
{
    sdr_func_init("");
    TEST_RUN(test_sdr_acc_cpu);
    TEST_RUN(test_sdr_acc_plugin);
    return 0;
}
