//
//  Acquisition benchmark and regression on recorded IF data.
//
//  usage: bench_sdr_acq [-fmt RAW32|INT8X2] [-f fs_MHz] [-fo f1,f2,...]
//             [-toff s] [-n nrep] [-acc backend] [-wisdom file] [-o out_file]
//             -sig sig -prn prns -rfch ch [-sig ...] file
//
//  -wisdom file: FFTW wisdom file (FFTW build). If the file does not exist,
//  wisdom of the DFT sizes of the signals is generated (FFTW_PATIENT) and saved.
//
//  Each signal search runs the receiver channel state machine (sdr_ch_update()
//  in SDR_STATE_SRCH) over the same IF samples and reports detection results,
//  a hash of the correlation-power map and timing/utilization statistics.
//
#include <sys/resource.h>
#include <time.h>
#include "pocket_sdr.h"

#define MAX_SIG   64
#define T_DATA    0.2       // IF data length read per RF channel (s)

typedef struct {            // signal spec
    char sig[16];
    int prns[SDR_MAX_NPRN], nprn, rfch;
} sig_spec_t;

typedef struct {            // search result
    char sig[16];
    int prn, found, idx[2];
    double cn0, fd, coff, P_max;
    uint64_t hash;
    double t_wall, t_cpu, t_npu, energy;
} acq_res_t;

// wall clock (s) --------------------------------------------------------------
static double wall_time(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// process CPU time (s) --------------------------------------------------------
static double cpu_time(void)
{
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    return ru.ru_utime.tv_sec + ru.ru_utime.tv_usec * 1e-6 +
        ru.ru_stime.tv_sec + ru.ru_stime.tv_usec * 1e-6;
}

// read uint64 from sysfs file (-1: error) -------------------------------------
static double read_sysfs(const char *path)
{
    FILE *fp = fopen(path, "r");
    double val = -1.0;
    if (fp) {
        if (fscanf(fp, "%lf", &val) != 1) val = -1.0;
        fclose(fp);
    }
    return val;
}

// NPU busy time (s) (Intel NPU sysfs, -1: not available) ----------------------
static double npu_busy(void)
{
    double v = read_sysfs("/sys/class/accel/accel0/device/npu_busy_time_us");
    return v < 0.0 ? -1.0 : v * 1e-6;
}

// package energy (J) (RAPL, -1: not available) --------------------------------
static double pkg_energy(void)
{
    double v = read_sysfs("/sys/class/powercap/intel-rapl:0/energy_uj");
    return v < 0.0 ? -1.0 : v * 1e-6;
}

// FNV-1a hash of float array ----------------------------------------------------
static uint64_t hash_float(const float *P, int n)
{
    uint64_t h = 1469598103934665603ULL;
    const uint8_t *p = (const uint8_t *)P;
    for (size_t i = 0; i < sizeof(float) * n; i++) {
        h = (h ^ p[i]) * 1099511628211ULL;
    }
    return h;
}

// read RF channel IF data (Pocket SDR FE raw / INT8X2) --------------------------
static sdr_buff_t *read_if(const char *file, int fmt, double fs, int rfch,
    double toff, double tlen)
{
    static const int8_t valI[] = {1, 3, -1, -3}, valQ[] = {-1, -3, 1, 3};
    int bs = fmt == SDR_FMT_RAW32 ? 4 : (fmt == SDR_FMT_RAW16 ? 2 : 2);
    int N = (int)(fs * tlen);
    FILE *fp = fopen(file, "rb");
    if (!fp) {
        fprintf(stderr, "file open error %s\n", file);
        return NULL;
    }
    uint8_t *raw = (uint8_t *)sdr_malloc(bs * N);
    fseeko(fp, (off_t)(fs * toff) * bs, SEEK_SET);
    if (fread(raw, bs, N, fp) < (size_t)N) {
        fprintf(stderr, "file read error %s\n", file);
        fclose(fp);
        sdr_free(raw);
        return NULL;
    }
    fclose(fp);
    sdr_buff_t *buff = sdr_buff_new(N, 2);
    for (int i = 0; i < N; i++) {
        if (fmt == SDR_FMT_INT8X2) {
            int8_t I = (int8_t)raw[i*2], Q = (int8_t)raw[i*2+1];
            buff->data[i] = SDR_CPX8(I, -Q);
        } else { // same as gen_LUTC() for IQ-sampling 2 bit
            uint8_t b = raw[i * bs + rfch / 2];
            int8_t I = valI[(b >> (rfch % 2 ? 4 : 0)) & 0x3];
            int8_t Q = valQ[(b >> (rfch % 2 ? 6 : 2)) & 0x3];
            buff->data[i] = SDR_CPX8(I, Q);
        }
    }
    sdr_free(raw);
    return buff;
}

// search a signal by receiver channel state machine ---------------------------
static void search(const char *sig, int prn, const sdr_buff_t *buff, double fs,
    double fi, acq_res_t *res)
{
    sdr_ch_t *ch = sdr_ch_new(sig, prn, fs, fi);
    memset(res, 0, sizeof(acq_res_t));
    strcpy(res->sig, sig);
    res->prn = prn;
    if (!ch) return;

    // correlation power map of first T_ACQ by sdr_search_code() (same as search_sig())
    extern double sdr_t_acq;
    int M = 2 * ch->N, n = ch->acq->len_fds, ncall = 0;
    float *P = (float *)sdr_malloc(sizeof(float) * M * n);
    double t0 = wall_time(), c0 = cpu_time(), b0 = npu_busy(), e0 = pkg_energy();
    for (int ix = 0; ncall * ch->T < sdr_t_acq && ix + M <= buff->N; ix += ch->N) {
        sdr_search_code2(ch->acq->code_fft, ch->T, buff, ix, M, ch->N, fs,
            ch->fi, ch->acq->fds, n, P);
        ncall++;
    }
    res->t_wall = wall_time() - t0;
    res->t_cpu = cpu_time() - c0;
    res->t_npu = b0 < 0.0 ? -1.0 : npu_busy() - b0;
    res->energy = e0 < 0.0 ? -1.0 : pkg_energy() - e0;
    res->cn0 = sdr_corr_max(P, M, ch->N, n, ch->T, res->idx);
    res->P_max = P[res->idx[0] * M + res->idx[1]];
    res->fd = sdr_fine_dop(P, M, ch->acq->fds, n, res->idx);
    res->coff = res->idx[1] / fs;
    for (int i = 0; i < n; i++) { // hash of code phases used by search_sig()
        res->hash ^= hash_float(P + i * M, ch->N) + i;
    }
    sdr_free(P);

    // receiver channel state machine (SRCH -> LOCK/IDLE)
    ch->state = SDR_STATE_SRCH;
    for (int ix = 0; ch->state == SDR_STATE_SRCH && ix + M <= buff->N;
        ix += ch->N) {
        sdr_ch_update(ch, (ix + M) / fs, buff, ix);
    }
    res->found = ch->state == SDR_STATE_LOCK;
    if (res->found && (fabs(ch->fd - res->fd) > 1e-6 ||
        fabs(ch->coff - res->coff) > 1e-12)) {
        fprintf(stderr, "%s %d: state machine result mismatch\n", sig, prn);
    }
    sdr_ch_free(ch);
}

// generate FFTW wisdom of DFT sizes of signals if wisdom file not exist -------
static void gen_wisdom(const char *file, const sig_spec_t *sigs, int nsig,
    double fs)
{
#if defined(FFTW)
    int M[MAX_SIG], n = 0;
    FILE *fp;

    if (!*file || (fp = fopen(file, "r"))) {
        if (*file) fclose(fp);
        return;
    }
    for (int i = 0; i < nsig; i++) {
        int m = 2 * (int)(fs * sdr_code_cyc(sigs[i].sig) + 0.5), j;
        for (j = 0; j < n && M[j] != m; j++) ;
        if (j < n) continue;
        M[n++] = m;
        // wisdom is accumulated in FFTW and all exported to file
        fprintf(stderr, "generating FFTW wisdom (N=%d) ...\n", m);
        if (!sdr_gen_fftw_wisdom(file, m)) {
            fprintf(stderr, "FFTW wisdom generation error %s\n", file);
        }
    }
#endif
}

// main ------------------------------------------------------------------------
int main(int argc, char **argv)
{
    sig_spec_t sigs[MAX_SIG];
    int nsig = 0, fmt = SDR_FMT_RAW32, nrep = 1;
    double fs = 16e6, fo[SDR_MAX_RFCH] = {0}, toff = 1.0;
    const char *file = "", *out = "", *acc = "cpu", *wisdom = "";

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-fmt") && i + 1 < argc) {
            i++;
            fmt = !strcmp(argv[i], "INT8X2") ? SDR_FMT_INT8X2 : SDR_FMT_RAW32;
        } else if (!strcmp(argv[i], "-f") && i + 1 < argc) {
            fs = atof(argv[++i]) * 1e6;
        } else if (!strcmp(argv[i], "-fo") && i + 1 < argc) {
            sscanf(argv[++i], "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf", fo, fo + 1,
                fo + 2, fo + 3, fo + 4, fo + 5, fo + 6, fo + 7);
            for (int j = 0; j < SDR_MAX_RFCH; j++) fo[j] *= 1e6;
        } else if (!strcmp(argv[i], "-toff") && i + 1 < argc) {
            toff = atof(argv[++i]);
        } else if (!strcmp(argv[i], "-n") && i + 1 < argc) {
            nrep = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "-acc") && i + 1 < argc) {
            acc = argv[++i];
        } else if (!strcmp(argv[i], "-wisdom") && i + 1 < argc) {
            wisdom = argv[++i];
        } else if (!strcmp(argv[i], "-o") && i + 1 < argc) {
            out = argv[++i];
        } else if (!strcmp(argv[i], "-sig") && i + 1 < argc && nsig < MAX_SIG) {
            strcpy(sigs[nsig].sig, argv[++i]);
            sigs[nsig].nprn = 0;
            sigs[nsig++].rfch = 0;
        } else if (!strcmp(argv[i], "-prn") && i + 1 < argc && nsig > 0) {
            sigs[nsig-1].nprn = sdr_parse_nums(argv[++i], sigs[nsig-1].prns);
        } else if (!strcmp(argv[i], "-rfch") && i + 1 < argc && nsig > 0) {
            sigs[nsig-1].rfch = atoi(argv[++i]) - 1;
        } else {
            file = argv[i];
        }
    }
    gen_wisdom(wisdom, sigs, nsig, fs);
    sdr_func_init(wisdom);
    sdr_log_level(1);
    if (!sdr_acc_init(acc)) return -1;
#if defined(FFTW)
    printf("# backend: %s (FFT: FFTW%s)\n", acc, *wisdom ? " wisdom" : "");
#else
    printf("# backend: %s (FFT: PocketFFT)\n", acc);
#endif
    FILE *fp = *out ? fopen(out, "w") : NULL;

    for (int i = 0; i < nsig; i++) {
        sdr_buff_t *buff = read_if(file, fmt, fs, sigs[i].rfch, toff, T_DATA);
        if (!buff) return -1;
        double fi = sdr_sig_freq(sigs[i].sig) - fo[sigs[i].rfch];
        double tw = 0.0, tc = 0.0, tn = 0.0, te = 0.0;
        int nfound = 0;
        sdr_acc_stat(NULL, 1);

        for (int j = 0; j < sigs[i].nprn; j++) {
            acq_res_t res;
            for (int k = 0; k < nrep; k++) {
                search(sigs[i].sig, sigs[i].prns[j], buff, fs, fi, &res);
                tw += res.t_wall;
                tc += res.t_cpu;
                tn += res.t_npu;
                te += res.energy;
            }
            nfound += res.found;
            char line[256];
            snprintf(line, sizeof(line),
                "%-6s %3d %d %5.1f %4d %6d %9.2f %.7f %.6e %016llx",
                res.sig, res.prn, res.found, res.cn0, res.idx[0], res.idx[1],
                res.fd, res.coff * 1e3, res.P_max, (unsigned long long)res.hash);
            printf("%s %8.2f\n", line, res.t_wall * 1e3);
            if (fp) fprintf(fp, "%s\n", line);
        }
        int ns = sigs[i].nprn * nrep;
        printf("# %s: found=%d/%d search=%.2f ms/sig cpu=%.2f ms/sig "
            "cpu_util=%.1f%% npu_util=%.1f%% energy=%.3f J/sig\n", sigs[i].sig,
            nfound, sigs[i].nprn, tw / ns * 1e3, tc / ns * 1e3, tc / tw * 100.0,
            tn < 0.0 ? -1.0 : tn / tw * 100.0, te < 0.0 ? -1.0 : te / ns);
        sdr_acc_stat_t st;
        sdr_acc_stat(&st, 1);
        if (st.ncall > 0) {
            int nc = (int)st.ncall;
            printf("#   acc: calls=%d fallback=%d total=%.3f conv=%.3f in=%.3f "
                "exec=%.3f out=%.3f ms/call in=%.1f out=%.1f kB/call\n", nc,
                (int)st.nfall, st.t_total / nc * 1e3, st.t_conv / nc * 1e3,
                st.t_in / nc * 1e3, st.t_exec / nc * 1e3, st.t_out / nc * 1e3,
                st.bytes_in / nc / 1e3, st.bytes_out / nc / 1e3);
        }
        sdr_buff_free(buff);
    }
    if (fp) fclose(fp);
    sdr_acc_exit();
    return 0;
}
