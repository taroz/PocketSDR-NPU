//
//  Unit tests for sdr_acc.c (accelerator backends of parallel code search).
//
//  The plugin backend is tested if it is loadable (SDR_ACC_PATH or default
//  library path). SDR_ACC_TEST sets the plugin backends to be tested
//  (default: "ov:CPU,npu").
//
#include "test_sdr.h"

#define FS       4e6        // sampling frequency (Hz)
#define FI       0.0        // IF frequency (Hz)
#define PRN      5          // PRN number
#define COFF     1234       // code offset (samples)
#define DOP      1000.0     // Doppler frequency (Hz)
#define AMP      0.25       // signal amplitude (noise sigma = 1)
#define NCALL    10         // number of non-coherent integrations
#define TOL_NRMSE 0.05      // tolerance of normalized RMS error of powers
                            // (CPU reference has carrier LUT quantization)
#define TOL_CN0  0.3        // tolerance of C/N0 metric (dB)
#define TOL_FP16 0.005      // tolerance of NRMSE of FP16 backend vs FP32 backend

typedef struct {            // test data
    int N, M, len_fds, ncall;
    double T, fs;
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
static void gen_data(test_data_t *d, double fs, float fd_max, int ncall)
{
    int len_code;
    int8_t *code = sdr_gen_code("L1CA", PRN, &len_code);
    d->fs = fs;
    d->ncall = ncall;
    d->T = sdr_code_cyc("L1CA");
    d->N = (int)(fs * d->T);
    d->M = 2 * d->N;
    d->buff = sdr_buff_new(d->N * (ncall + 1), 2);
    srand(1);
    for (int i = 0; i < d->buff->N; i++) {
        double t = (double)(i - COFF) / fs;
        int j = (int)floor(t / d->T * len_code) % len_code;
        if (j < 0) j += len_code;
        double ph = 2.0 * PI * (FI + DOP) * i / fs;
        double I = AMP * code[j] * cos(ph) + randn();
        double Q = AMP * code[j] * sin(ph) + randn();
        d->buff->data[i] = SDR_CPX8(quant(I), quant(Q));
    }
    d->code_fft = sdr_cpx_malloc(d->M);
    sdr_gen_code_fft(code, NULL, len_code, d->T, 0.0, fs, d->N, d->N,
        d->code_fft);
    d->fds = sdr_dop_bins(d->T, 0.0f, fd_max, &d->len_fds);
}

// free test data --------------------------------------------------------------
static void free_data(test_data_t *d)
{
    sdr_buff_free(d->buff);
    sdr_cpx_free(d->code_fft);
    sdr_free(d->fds);
}

// non-coherent integrated correlation powers ----------------------------------
static float *search(const test_data_t *d, int cpu)
{
    float *P = (float *)calloc((size_t)d->M * d->len_fds, sizeof(float));
    for (int i = 0; i < d->ncall; i++) {
        if (cpu) {
            sdr_search_code_cpu(d->code_fft, d->T, d->buff, i * d->N, d->M,
                d->fs, FI, d->fds, d->len_fds, P);
        } else {
            sdr_search_code(d->code_fft, d->T, d->buff, i * d->N, d->M, d->fs,
                FI, d->fds, d->len_fds, P);
        }
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

// compare backend results with CPU reference (return powers) ------------------
static float *check_backend(const test_data_t *d, const float *P_ref,
    const char *name)
{
    int ix0[2], ix1[2];
    float *P = search(d, 0);
    float cn0_ref = sdr_corr_max(P_ref, d->M, d->N, d->len_fds, d->T, ix0);
    float cn0 = sdr_corr_max(P, d->M, d->N, d->len_fds, d->T, ix1);
    double e = nrmse(d, P_ref, P);
    printf("[%s nrmse=%.4f cn0=%.2f/%.2f] ", name, e, cn0_ref, cn0);
    TEST_ASSERT_EQ_INT(ix0[0], ix1[0]);
    TEST_ASSERT_EQ_INT(ix0[1], ix1[1]);
    TEST_ASSERT_TRUE(e < TOL_NRMSE);
    TEST_ASSERT_NEAR(cn0_ref, cn0, TOL_CN0);
    TEST_ASSERT_NEAR(sdr_fine_dop(P_ref, d->M, d->fds, d->len_fds, ix0),
        sdr_fine_dop(P, d->M, d->fds, d->len_fds, ix1), 0.05 * 0.5 / d->T);
    return P;
}

// CPU reference and dispatcher of CPU backend ---------------------------------
static void test_sdr_acc_cpu(void)
{
    test_data_t d;
    int ix[2];
    gen_data(&d, FS, 5000.0f, NCALL);
    TEST_ASSERT_EQ_INT(1, sdr_acc_init("cpu"));
    const char *name;
    TEST_ASSERT_EQ_INT(SDR_ACC_CPU, sdr_acc_type(&name));
    TEST_ASSERT_TRUE(!strcmp(name, "cpu"));
    float *P_ref = search(&d, 1);
    float *P = search(&d, 0);
    float cn0 = sdr_corr_max(P_ref, d.M, d.N, d.len_fds, d.T, ix);
    TEST_ASSERT_TRUE(cn0 > 40.0f);
    TEST_ASSERT_EQ_INT(COFF, ix[1]);
    TEST_ASSERT_NEAR(DOP, d.fds[ix[0]], 1e-3);
    TEST_ASSERT_TRUE(!memcmp(P_ref, P, sizeof(float) * d.M * d.len_fds));
    free(P_ref);
    free(P);
    free_data(&d);
}

// CPU optimized backend -------------------------------------------------------
static void test_sdr_acc_cpu_opt(void)
{
    test_data_t d;
    sdr_acc_stat_t stat;
    gen_data(&d, FS, 5000.0f, NCALL);
    float *P_ref = search(&d, 1);
    TEST_ASSERT_EQ_INT(1, sdr_acc_init("cpu_opt"));
    TEST_ASSERT_EQ_INT(SDR_ACC_CPUOPT, sdr_acc_type(NULL));
    sdr_acc_stat(NULL, 1);
    free(check_backend(&d, P_ref, "cpu_opt"));
    sdr_acc_stat(&stat, 1);
    TEST_ASSERT_EQ_INT(NCALL, (int)stat.ncall);

    // non-integer Doppler bin step -> CPU fallback (identical to reference)
    float fds[3] = {-300.0f, 10.0f, 320.0f};
    float *P0 = (float *)calloc(d.M * 3, sizeof(float));
    float *P1 = (float *)calloc(d.M * 3, sizeof(float));
    sdr_search_code_cpu(d.code_fft, d.T, d.buff, 0, d.M, FS, FI, fds, 3, P0);
    sdr_search_code(d.code_fft, d.T, d.buff, 0, d.M, FS, FI, fds, 3, P1);
    TEST_ASSERT_TRUE(!memcmp(P0, P1, sizeof(float) * d.M * 3));
    sdr_acc_stat(&stat, 1);
    TEST_ASSERT_EQ_INT(1, (int)stat.nfall);
    sdr_acc_exit();
    TEST_ASSERT_EQ_INT(SDR_ACC_CPU, sdr_acc_type(NULL));
    free(P0);
    free(P1);
    free(P_ref);
    free_data(&d);
}

// plugin backends (skipped if not loadable) -----------------------------------
static void test_sdr_acc_plugin(void)
{
    const char *env = getenv("SDR_ACC_TEST");
    char list[256], *p, *q;
    test_data_t d;
    float *P_fp32 = NULL, *P_npu = NULL;
    gen_data(&d, FS, 5000.0f, NCALL);
    float *P_ref = search(&d, 1);

    // unknown plugin -> error and CPU reference backend
    TEST_ASSERT_EQ_INT(0, sdr_acc_init("no_such_plugin"));
    TEST_ASSERT_EQ_INT(SDR_ACC_CPU, sdr_acc_type(NULL));

    snprintf(list, sizeof(list), "%s", env ? env : "ov:CPU,npu");
    for (p = list; p && *p; p = q) {
        if ((q = strchr(p, ','))) *q++ = '\0';
        if (!sdr_acc_init(p)) {
            printf("[%s SKIP] ", p);
            continue;
        }
        TEST_ASSERT_EQ_INT(SDR_ACC_PLUGIN, sdr_acc_type(NULL));
        sdr_acc_stat(NULL, 1);
        float *P = check_backend(&d, P_ref, p);
        if (!strcmp(p, "ov:CPU")) P_fp32 = P;
        else if (!strcmp(p, "npu")) P_npu = P;
        else free(P);

        // unsupported DFT size (no factorization) -> CPU fallback
        float P0[7] = {0}, P1[7] = {0};
        sdr_cpx_t code[7] = {{1, 0}};
        sdr_search_code_cpu(code, d.T, d.buff, 0, 7, FS, FI, d.fds, 1, P0);
        sdr_search_code(code, d.T, d.buff, 0, 7, FS, FI, d.fds, 1, P1);
        TEST_ASSERT_TRUE(!memcmp(P0, P1, sizeof(P0)));
        sdr_acc_stat_t stat;
        sdr_acc_stat(&stat, 1);
        TEST_ASSERT_EQ_INT(NCALL, (int)stat.ncall);
        TEST_ASSERT_EQ_INT(1, (int)stat.nfall);
        sdr_acc_exit();
    }
    // FP16 (NPU) vs FP32 (OpenVINO CPU) of the same tensor graph
    if (P_fp32 && P_npu) {
        double e = nrmse(&d, P_fp32, P_npu);
        printf("[npu vs ov:CPU nrmse=%.5f] ", e);
        TEST_ASSERT_TRUE(e < TOL_FP16);
    }
    free(P_fp32);
    free(P_npu);
    free(P_ref);
    free_data(&d);
}

// plugin backends with Doppler bins processed in chunks ------------------------
//   (fs = 16 MHz, +/-21 kHz: 85 bins x DFT size 32000 > max chunk size of the
//   OpenVINO plugin, skipped if not loadable)
static void test_sdr_acc_chunk(void)
{
    const char *env = getenv("SDR_ACC_TEST");
    char list[256], *p, *q;
    test_data_t d;
    gen_data(&d, 16e6, 21000.0f, 2);
    float *P_ref = search(&d, 1);
    TEST_ASSERT_EQ_INT(85, d.len_fds);

    snprintf(list, sizeof(list), "%s", env ? env : "ov:CPU,npu");
    for (p = list; p && *p; p = q) {
        if ((q = strchr(p, ','))) *q++ = '\0';
        if (!sdr_acc_init(p)) {
            printf("[%s SKIP] ", p);
            continue;
        }
        sdr_acc_stat_t stat;
        sdr_acc_stat(NULL, 1);
        free(check_backend(&d, P_ref, p));
        sdr_acc_stat(&stat, 1);
        TEST_ASSERT_EQ_INT(d.ncall, (int)stat.ncall);
        TEST_ASSERT_EQ_INT(0, (int)stat.nfall);
        sdr_acc_exit();
    }
    free(P_ref);
    free_data(&d);
}

// batched tracking correlators -----------------------------------------------
//   cpu, cpu_mt: identical to sdr_corr_std(), plugins: relative error < 0.05
//   (jobs with 2 IF sample blocks and across the IF buffer boundary)
#define NJOB 40             // number of correlator jobs
#define SQR(x)   ((x) * (x))

static void test_sdr_acc_corr_batch(void)
{
    const char *env = getenv("SDR_ACC_TEST");
    char list[256], *p, *q;
    test_data_t d;
    static sdr_ch_t *chs[NJOB];
    static sdr_corr_job_t jobs[NJOB], refs[NJOB];
    static sdr_cpx_t corr[NJOB][SDR_MAX_CORR], corr_ref[NJOB][SDR_MAX_CORR];
    gen_data(&d, FS, 5000.0f, 2);
    
    for (int i = 0; i < NJOB; i++) {
        chs[i] = sdr_ch_new("L1CA", i % 32 + 1, FS, FI);
        if (i % 4 == 3) sdr_ch_set_corr(chs[i], 7, 1e-6); // extra taps
        sdr_corr_job_t *job = jobs + i;
        job->buff = d.buff;
        job->ix = i < NJOB / 2 ? 0 : d.buff->N - d.N / 3; // across boundary
        job->N = d.N;
        job->fs = FS;
        job->fc = DOP + (i - NJOB / 2) * 37.0;
        job->phi = i * 0.031;
        job->code = chs[i]->trk->code;
        job->code_sum = chs[i]->trk->code_sum;
        job->scale = chs[i]->trk->code_scale;
        job->coff = i == 0 ? COFF : fmod(i * 731.37, d.N);
        job->pos = chs[i]->trk->pos;
        job->npos = chs[i]->trk->npos + chs[i]->trk->nposx;
        job->corr = corr[i];
        refs[i] = *job;
        refs[i].corr = corr_ref[i];
        sdr_corr_std(job->buff, job->ix, job->N, job->fs, job->fc, job->phi,
            job->code, job->code_sum, NULL, NULL, job->scale, job->coff,
            job->pos, job->npos, 0, refs[i].corr, refs[i].C);
    }
    snprintf(list, sizeof(list), "cpu,cpu_mt:4,%s", env ? env : "ov:CPU,npu");
    for (p = list; p && *p; p = q) {
        if ((q = strchr(p, ','))) *q++ = '\0';
        if (!sdr_acc_trk_init(p)) {
            printf("[%s SKIP] ", p);
            continue;
        }
        int exact = !strncmp(p, "cpu", 3);
        sdr_acc_stat_t stat;
        sdr_acc_trk_stat(NULL, 1);
        memset(corr, 0, sizeof(corr));
        sdr_acc_corr_batch(jobs, NJOB);
        sdr_acc_trk_stat(&stat, 1);
        TEST_ASSERT_EQ_INT(NJOB, (int)stat.njob);
        TEST_ASSERT_EQ_INT(0, (int)stat.nfall);
        double err[NJOB], med;
        for (int i = 0; i < NJOB; i++) {
            if (exact) {
                TEST_ASSERT_TRUE(!memcmp(corr[i], corr_ref[i],
                    sizeof(sdr_cpx_t) * jobs[i].npos));
                TEST_ASSERT_TRUE(!memcmp(jobs[i].C, refs[i].C,
                    sizeof(sdr_cpx_t) * 2));
            }
            double se = 0.0, ss = 0.0;
            for (int j = 0; j < jobs[i].npos; j++) {
                for (int k = 0; k < 2; k++) {
                    se += SQR(corr[i][j][k] - corr_ref[i][j][k]);
                    ss += SQR(corr_ref[i][j][k]);
                }
            }
            err[i] = sqrt(se / ss);
        }
        double e0, s0 = 0.0, d0 = 0.0; // signal taps (P, E, L) of job 0
        for (int j = 0; j < 3; j++) for (int k = 0; k < 2; k++) {
            d0 += SQR(corr[0][j][k] - corr_ref[0][j][k]);
            s0 += SQR(corr_ref[0][j][k]);
        }
        e0 = sqrt(d0 / s0);
        for (int i = 0; i < NJOB; i++) for (int j = i + 1; j < NJOB; j++) {
            if (err[j] < err[i]) {double t = err[i]; err[i] = err[j]; err[j] = t;}
        }
        med = err[NJOB / 2];
        // P correlator of the signal (job 0)
        double P0 = sqrt(SQR(corr[0][0][0]) + SQR(corr[0][0][1]));
        double P1 = sqrt(SQR(corr_ref[0][0][0]) + SQR(corr_ref[0][0][1]));
        printf("[%s err(signal)=%.4f err(median)=%.4f P=%.4f/%.4f] ", p, e0, med,
            P1, P0);
        // (noise-only jobs: relative errors by the carrier LUT quantization of
        // the CPU reference and ambiguous polarity over the code wrap-around)
        TEST_ASSERT_TRUE(e0 < 0.05);
        TEST_ASSERT_TRUE(med < 0.1);
        TEST_ASSERT_NEAR(P1, P0, 0.02 * P1);
        sdr_acc_trk_exit();
    }
    TEST_ASSERT_EQ_INT(SDR_ACC_CPU, sdr_acc_trk_type(NULL));
    for (int i = 0; i < NJOB; i++) sdr_ch_free(chs[i]);
    free_data(&d);
}

int main(void)
{
    sdr_func_init("");
    TEST_RUN(test_sdr_acc_cpu);
    TEST_RUN(test_sdr_acc_cpu_opt);
    TEST_RUN(test_sdr_acc_plugin);
    TEST_RUN(test_sdr_acc_chunk);
    TEST_RUN(test_sdr_acc_corr_batch);
    return 0;
}
