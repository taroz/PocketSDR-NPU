//
//  Batched tracking correlator benchmark (scaling by channels and taps).
//
//  usage: bench_sdr_corr [-sig sig] [-f fs_MHz] [-nch n[,n...]]
//             [-ntap n[,n...]] [-tacc backend[,backend...]] [-n nrep] [file]
//
//  The correlators of nch channels (PRNs of sig, random carrier and code
//  offsets) with ntap taps are executed by sdr_acc_corr_batch() on the same
//  IF samples (file: IF data INT8X2 or random) and the time per batch is
//  measured for each backend. A batch within the code period (real-time)
//  is marked with "*". The outputs of the backends are compared with the CPU
//  reference (relative error of correlations).
//
#include <math.h>
#include <sys/resource.h>
#include <time.h>
#include "pocket_sdr.h"

#define MAX_NCH  4096
#define MIN(x, y) ((x) < (y) ? (x) : (y))
#define MAX(x, y) ((x) > (y) ? (x) : (y))

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

// read sysfs value (-1: error) ------------------------------------------------
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

// parse comma-separated list --------------------------------------------------
static int parse_list(const char *str, char list[][64], int max)
{
    int n = 0;
    char buff[1024], *p, *q;
    snprintf(buff, sizeof(buff), "%s", str);
    for (p = buff; p && *p && n < max; p = q) {
        if ((q = strchr(p, ','))) *q++ = '\0';
        snprintf(list[n++], 64, "%s", p);
    }
    return n;
}

// main ------------------------------------------------------------------------
int main(int argc, char **argv)
{
    static sdr_ch_t *chs[MAX_NCH];
    static sdr_corr_job_t jobs[MAX_NCH], refs[MAX_NCH];
    static sdr_cpx_t corr[MAX_NCH][SDR_MAX_CORR], corr_ref[MAX_NCH][SDR_MAX_CORR];
    char nchs[16][64], ntaps[16][64], taccs[16][64];
    int nn = parse_list("16,32,64,128,256,512", nchs, 16);
    int nt = parse_list("4", ntaps, 16), nb = parse_list("cpu,cpu_mt,npu", taccs, 16);
    int nrep = 50;
    double fs = 16e6;
    const char *sig = "L1CA", *file = "";

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-sig") && i + 1 < argc) {
            sig = argv[++i];
        } else if (!strcmp(argv[i], "-f") && i + 1 < argc) {
            fs = atof(argv[++i]) * 1e6;
        } else if (!strcmp(argv[i], "-nch") && i + 1 < argc) {
            nn = parse_list(argv[++i], nchs, 16);
        } else if (!strcmp(argv[i], "-ntap") && i + 1 < argc) {
            nt = parse_list(argv[++i], ntaps, 16);
        } else if (!strcmp(argv[i], "-tacc") && i + 1 < argc) {
            nb = parse_list(argv[++i], taccs, 16);
        } else if (!strcmp(argv[i], "-n") && i + 1 < argc) {
            nrep = atoi(argv[++i]);
        } else {
            file = argv[i];
        }
    }
    sdr_func_init("");
    sdr_log_level(1);

    // IF samples (2 code periods)
    double T = sdr_code_cyc(sig);
    int N = (int)(fs * T);
    sdr_buff_t *buff = sdr_buff_new(2 * N, 2);
    FILE *fp = *file ? fopen(file, "rb") : NULL;
    srand(1);
    for (int i = 0; i < buff->N; i++) {
        int8_t I, Q;
        if (fp && fread(&I, 1, 1, fp) == 1 && fread(&Q, 1, 1, fp) == 1) {
            buff->data[i] = SDR_CPX8(I, -Q);
        } else {
            buff->data[i] = SDR_CPX8(rand() % 7 - 3, rand() % 7 - 3);
        }
    }
    if (fp) fclose(fp);

    int nmax = 0;
    for (int i = 0; i < nn; i++) nmax = MAX(nmax, atoi(nchs[i]));
    nmax = MIN(nmax, MAX_NCH);
    for (int i = 0; i < nmax; i++) {
        chs[i] = sdr_ch_new(sig, i % 32 + 1, fs, 0.0);
    }
    printf("# sig=%s fs=%.3f MHz N=%d T=%.0f ms (real-time: batch < %.3f ms)\n",
        sig, fs * 1e-6, N, T * 1e3, T * 1e3);
    printf("# %-10s %5s %5s %10s %10s %10s %8s %8s %10s\n", "backend", "nch",
        "ntap", "batch(ms)", "us/ch", "cpu(ms)", "npu(%)", "energy", "rel.err");
    printf("#   (energy: J/ch-batch, conv/in/exec: ms/batch of generic conversion,"
        " backend input and execution)\n");

    for (int b = 0; b < nb; b++) {
        if (!sdr_acc_trk_init(taccs[b])) continue;
        const char *name;
        sdr_acc_trk_type(&name);
        for (int t = 0; t < nt; t++) {
            int ntap = atoi(ntaps[t]);
            for (int i = 0; i < nmax; i++) { // taps: P, E, L, N + extra
                sdr_ch_set_corr(chs[i], MAX(0, ntap - 4), 2.0 * T /
                    chs[i]->len_code);
            }
            for (int k = 0; k < nn; k++) {
                int nch = MIN(atoi(nchs[k]), nmax);
                for (int i = 0; i < nch; i++) {
                    sdr_ch_t *ch = chs[i];
                    sdr_corr_job_t *job = jobs + i;
                    job->buff = buff;
                    job->ix = 0;
                    job->N = N;
                    job->fs = fs;
                    job->fc = (i * 0.37 - 0.5) * 8000.0 + 0.123;
                    job->phi = i * 0.0137;
                    job->code = ch->trk->code;
                    job->code_sum = ch->trk->code_sum;
                    job->scale = ch->trk->code_scale;
                    job->coff = fmod(i * 1234.567, N);
                    job->pos = ch->trk->pos;
                    job->npos = ch->trk->npos + ch->trk->nposx;
                    job->corr = corr[i];
                    refs[i] = *job;
                    refs[i].corr = corr_ref[i];
                    sdr_corr_std(buff, 0, N, fs, job->fc, job->phi, job->code,
                        job->code_sum, NULL, NULL, job->scale, job->coff,
                        job->pos, job->npos, 0, refs[i].corr, refs[i].C);
                }
                sdr_acc_corr_batch(jobs, nch); // warm-up (model compile)
                sdr_acc_trk_stat(NULL, 1);
                double e0 = read_sysfs("/sys/class/powercap/intel-rapl:0/energy_uj");
                double n0 = read_sysfs("/sys/class/accel/accel0/device/npu_busy_time_us");
                double t0 = wall_time(), c0 = cpu_time();
                for (int r = 0; r < nrep; r++) {
                    sdr_acc_corr_batch(jobs, nch);
                }
                double tw = (wall_time() - t0) / nrep, tc = (cpu_time() - c0) / nrep;
                double e1 = read_sysfs("/sys/class/powercap/intel-rapl:0/energy_uj");
                double n1 = read_sysfs("/sys/class/accel/accel0/device/npu_busy_time_us");
                double en = e0 < 0.0 ? -1.0 : (e1 - e0) * 1e-6 / nrep / nch;
                double nu = n0 < 0.0 ? -1.0 : (n1 - n0) * 1e-6 / nrep / tw * 100.0;
                double se = 0.0, ss = 0.0;
                for (int i = 0; i < nch; i++) {
                    for (int j = 0; j < jobs[i].npos; j++) {
                        for (int m = 0; m < 2; m++) {
                            double d = jobs[i].corr[j][m] - refs[i].corr[j][m];
                            se += d * d;
                            ss += refs[i].corr[j][m] * refs[i].corr[j][m];
                        }
                    }
                }
                sdr_acc_stat_t st;
                sdr_acc_trk_stat(&st, 1);
                double nc = st.ncall > 0 ? (double)st.ncall : 1.0;
                printf("  %-10s %5d %5d %10.3f%s %9.2f %10.3f %8.1f %8.2e %10.5f"
                    " %7.3f %7.3f %7.3f\n", name, nch, jobs[0].npos, tw * 1e3,
                    tw <= T ? "*" : " ", tw / nch * 1e6, tc * 1e3, nu, en,
                    sqrt(se / ss), st.t_conv / nc * 1e3, st.t_in / nc * 1e3,
                    st.t_exec / nc * 1e3);
                fflush(stdout);
            }
        }
        sdr_acc_trk_exit();
    }
    for (int i = 0; i < nmax; i++) sdr_ch_free(chs[i]);
    sdr_buff_free(buff);
    return 0;
}
