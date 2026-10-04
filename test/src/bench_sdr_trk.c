//
//  Tracking benchmark and regression on recorded IF data.
//
//  usage: bench_sdr_trk [-fmt RAW32|INT8X2] [-f fs_MHz] [-fo f1,f2,...]
//             [-toff s] [-tlen s] [-acc backend] [-tacc backend] [-batch]
//             [-nposx n] [-o out_file] -sig sig -prn prns -rfch ch
//             [-sig ...] file
//
//  The signals are acquired by the receiver channel state machine
//  (sdr_ch_update() in SDR_STATE_SRCH) and tracked over the same IF samples
//  for tlen seconds. Without -batch, the channels are updated by
//  sdr_ch_update() (original). With -batch, the channels of the same RF
//  channel and the same code period are updated epoch by epoch with the
//  tracking correlator separated (sdr_ch_update_pre(), sdr_acc_corr_batch()
//  by the tracking backend -tacc, sdr_ch_update_post()). The tracking states
//  and the correlator outputs of each epoch are written to out_file for the
//  regression (cmp_trk.py).
//
#include <sys/resource.h>
#include <time.h>
#include "pocket_sdr.h"

#define MAX_SIG   64
#define MAX_CH    1024

typedef struct {            // signal spec
    char sig[16];
    int prns[SDR_MAX_NPRN], nprn, rfch;
} sig_spec_t;

typedef struct {            // tracking channel
    sdr_ch_t *ch;           // receiver channel
    int rfch;               // RF channel
    int ix0;                // start index of tracking (samples)
} trk_ch_t;

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

// read RF channel IF data (Pocket SDR FE raw / INT8X2) --------------------------
static sdr_buff_t *read_if(const char *file, int fmt, double fs, int rfch,
    double toff, double tlen)
{
    static const int8_t valI[] = {1, 3, -1, -3}, valQ[] = {-1, -3, 1, 3};
    int bs = fmt == SDR_FMT_RAW32 ? 4 : 2, nblk = 1 << 20;
    int64_t N = (int64_t)(fs * tlen);
    FILE *fp = fopen(file, "rb");
    if (!fp) {
        fprintf(stderr, "file open error %s\n", file);
        return NULL;
    }
    uint8_t *raw = (uint8_t *)sdr_malloc(bs * nblk);
    fseeko(fp, (off_t)(fs * toff) * bs, SEEK_SET);
    sdr_buff_t *buff = sdr_buff_new((int)N, 2);
    for (int64_t i0 = 0; i0 < N; i0 += nblk) {
        int n = (int)(N - i0 < nblk ? N - i0 : nblk);
        if (fread(raw, bs, n, fp) < (size_t)n) {
            fprintf(stderr, "file read error %s\n", file);
            fclose(fp);
            sdr_free(raw);
            sdr_buff_free(buff);
            return NULL;
        }
        for (int i = 0; i < n; i++) {
            uint8_t *d = buff->data + i0 + i;
            if (fmt == SDR_FMT_INT8X2) {
                int8_t I = (int8_t)raw[i*2], Q = (int8_t)raw[i*2+1];
                *d = SDR_CPX8(I, -Q);
            } else { // same as gen_LUTC() for IQ-sampling 2 bit
                uint8_t b = raw[i * bs + rfch / 2];
                int8_t I = valI[(b >> (rfch % 2 ? 4 : 0)) & 0x3];
                int8_t Q = valQ[(b >> (rfch % 2 ? 6 : 2)) & 0x3];
                *d = SDR_CPX8(I, Q);
            }
        }
    }
    fclose(fp);
    sdr_free(raw);
    return buff;
}

// acquire signal by receiver channel state machine ----------------------------
static int acquire(sdr_ch_t *ch, const sdr_buff_t *buff, double tmax)
{
    int M = 2 * ch->N;
    ch->state = SDR_STATE_SRCH;
    for (int ix = 0; ch->state == SDR_STATE_SRCH && ix + M <= buff->N &&
        ix < tmax * ch->fs; ix += ch->N) {
        sdr_ch_update(ch, (double)(ix + M) / ch->fs, buff, ix);
        if (ch->state == SDR_STATE_LOCK) return ix + ch->N;
    }
    return -1;
}

// output tracking state of channel at epoch -----------------------------------
static void out_state(FILE *fp, const sdr_ch_t *ch, int ix)
{
    if (!fp || ch->state != SDR_STATE_LOCK) return;
    int npos = ch->trk->npos + ch->trk->nposx;
    fprintf(fp, "%-6s %3d %9d %6d %.9e %.9e %.9e %.9e %.6f %d %d %d", ch->sig,
        ch->prn, ix, ch->lock, ch->coff, ch->fd, ch->adr, ch->phi, ch->cn0,
        ch->trk->sec_sync, ch->nav->ssync, ch->nav->fsync);
    for (int i = 0; i < npos; i++) {
        fprintf(fp, " %.7e %.7e", ch->trk->C[i][0], ch->trk->C[i][1]);
    }
    fprintf(fp, " %.7e %.7e\n", ch->trk->P[SDR_N_HIST-1][0],
        ch->trk->P[SDR_N_HIST-1][1]);
}

// main ------------------------------------------------------------------------
int main(int argc, char **argv)
{
    static trk_ch_t chs[MAX_CH];
    static sdr_corr_job_t jobs[MAX_CH];
    static sdr_ch_t *jch[MAX_CH];
    sig_spec_t sigs[MAX_SIG];
    sdr_buff_t *buffs[SDR_MAX_RFCH] = {0};
    int nsig = 0, fmt = SDR_FMT_RAW32, batch = 0, nposx = 0, nch = 0;
    double fs = 16e6, fo[SDR_MAX_RFCH] = {0}, toff = 1.0, tlen = 3.0;
    double tacq = 1.0;
    const char *file = "", *out = "", *acc = "cpu_opt", *tacc = "cpu";

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
        } else if (!strcmp(argv[i], "-tlen") && i + 1 < argc) {
            tlen = atof(argv[++i]);
        } else if (!strcmp(argv[i], "-acc") && i + 1 < argc) {
            acc = argv[++i];
        } else if (!strcmp(argv[i], "-tacc") && i + 1 < argc) {
            tacc = argv[++i];
        } else if (!strcmp(argv[i], "-batch")) {
            batch = 1;
        } else if (!strcmp(argv[i], "-nposx") && i + 1 < argc) {
            nposx = atoi(argv[++i]);
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
    sdr_func_init("");
    sdr_log_level(1);
    if (!sdr_acc_init(acc) || !sdr_acc_trk_init(tacc)) return -1;
    const char *tname;
    sdr_acc_trk_type(&tname);
    printf("# mode: %s, tracking backend: %s\n", batch ? "batch" : "orig",
        batch ? tname : "-");

    // read IF data and acquire signals
    for (int i = 0; i < nsig; i++) {
        int rfch = sigs[i].rfch;
        if (!buffs[rfch] &&
            !(buffs[rfch] = read_if(file, fmt, fs, rfch, toff, tacq + tlen))) {
            return -1;
        }
        double fi = sdr_sig_freq(sigs[i].sig) - fo[rfch];
        for (int j = 0; j < sigs[i].nprn && nch < MAX_CH; j++) {
            sdr_ch_t *ch = sdr_ch_new(sigs[i].sig, sigs[i].prns[j], fs, fi);
            if (!ch) continue;
            if (nposx > 0) sdr_ch_set_corr(ch, nposx, 2.0 * ch->T / ch->len_code);
            int ix0 = acquire(ch, buffs[rfch], tacq);
            if (ix0 < 0) {
                sdr_ch_free(ch);
                continue;
            }
            chs[nch].ch = ch;
            chs[nch].rfch = rfch;
            chs[nch++].ix0 = ix0;
        }
    }
    printf("# tracking channels: %d\n", nch);
    FILE *fp = *out ? fopen(out, "w") : NULL;

    // track signals by groups of RF channel and code period
    double t_wall = 0.0, t_cpu = 0.0, t_corr = 0.0;
    int64_t nepoch = 0;
    int done[MAX_CH] = {0};
    sdr_acc_trk_stat(NULL, 1);
    for (int i = 0; i < nch; i++) {
        if (done[i]) continue;
        int rfch = chs[i].rfch, N = chs[i].ch->N, ng = 0, gch[MAX_CH];
        int ix_min = chs[i].ix0;
        for (int j = i; j < nch; j++) {
            if (done[j] || chs[j].rfch != rfch || chs[j].ch->N != N) continue;
            gch[ng++] = j;
            done[j] = 1;
            if (chs[j].ix0 < ix_min) ix_min = chs[j].ix0;
        }
        const sdr_buff_t *buff = buffs[rfch];
        double t0 = wall_time(), c0 = cpu_time();
        for (int ix = ix_min; ix + 2 * N <= buff->N; ix += N) {
            double time = (double)(ix + 2 * N) / fs;
            int njob = 0;
            for (int k = 0; k < ng; k++) {
                trk_ch_t *c = chs + gch[k];
                if (ix < c->ix0) continue;
                nepoch++;
                if (!batch) {
                    sdr_ch_update(c->ch, time, buff, ix);
                } else if (sdr_ch_update_pre(c->ch, time, buff, ix,
                        jobs + njob)) {
                    jch[njob++] = c->ch;
                }
            }
            if (batch) {
                double t1 = wall_time();
                sdr_acc_corr_batch(jobs, njob);
                t_corr += wall_time() - t1;
                for (int k = 0; k < njob; k++) {
                    sdr_ch_update_post(jch[k], jobs + k);
                }
            }
            for (int k = 0; k < ng; k++) {
                trk_ch_t *c = chs + gch[k];
                if (ix >= c->ix0) out_state(fp, c->ch, ix);
            }
        }
        t_wall += wall_time() - t0;
        t_cpu += cpu_time() - c0;
    }
    if (fp) fclose(fp);

    int nlock = 0;
    for (int i = 0; i < nch; i++) nlock += chs[i].ch->state == SDR_STATE_LOCK;
    printf("# channel-epochs=%lld locked(end)=%d/%d time=%.3f s cpu=%.3f s "
        "(%.2f us/ch-epoch)\n", (long long)nepoch, nlock, nch, t_wall, t_cpu,
        t_wall / nepoch * 1e6);
    if (batch) {
        sdr_acc_stat_t st;
        sdr_acc_trk_stat(&st, 1);
        printf("#   corr: batches=%lld jobs=%lld fallback=%lld time=%.3f s "
            "(%.2f us/job) conv=%.3f in=%.3f exec=%.3f out=%.3f s\n",
            (long long)st.ncall, (long long)st.njob, (long long)st.nfall,
            t_corr, st.njob > 0 ? t_corr / st.njob * 1e6 : 0.0, st.t_conv,
            st.t_in, st.t_exec, st.t_out);
    }
    for (int i = 0; i < nch; i++) sdr_ch_free(chs[i].ch);
    for (int i = 0; i < SDR_MAX_RFCH; i++) sdr_buff_free(buffs[i]);
    sdr_acc_trk_exit();
    sdr_acc_exit();
    return 0;
}
