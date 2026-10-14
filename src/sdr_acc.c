//
//  Pocket SDR C Library - GNSS SDR Accelerator Backend Functions.
//
//  Generic accelerator interface between the GNSS receiver logic and the
//  processing backends. The CPU reference implementations are kept in
//  sdr_func.c. Hardware-specific backends are loaded as plugins (sdr_acc.h).
//
//    backend   description
//    cpu       CPU reference (sdr_search_code_cpu()) (default)
//    cpu_opt   CPU optimized (one DFT of IF data shared by Doppler bins)
//    npu       NPU by plugin ov (OpenVINO) on device NPU (= ov:NPU)
//    <p>[:opt] plugin libsdr_acc_<p>.so (.dll) with option string opt
//
//  Tracking correlator backends (sdr_acc_trk_init(), sdr_acc_corr_batch()):
//
//    backend   description
//    cpu       CPU reference (sdr_corr_std() for each job) (default)
//    cpu_mt[:n] CPU multi-threaded (n threads, default: number of CPU cores of
//              the thread affinity of the caller)
//    npu       NPU by plugin ov (OpenVINO) on device NPU (= ov:NPU)
//    <p>[:opt] plugin libsdr_acc_<p>.so (.dll) with option string opt
//
//  Author:
//  T.SUZUKI
//
//  History:
//  2026-10-02  1.0  new
//  2026-10-03  1.1  add tracking correlator backends (batched correlator)
//  2026-10-08  1.2  cpu_mt: default threads = CPUs of the thread affinity
//  2026-10-11  1.3  add sdr_acc_search_multi(), sdr_acc_search_code_multi()
//                   plugin: CPU optimized for unsupported shapes (counted
//                   as fallbacks)
//
#if !defined(WIN32) && !defined(_GNU_SOURCE)
#define _GNU_SOURCE         // sched_getaffinity()
#endif
#include <math.h>
#include "pocket_sdr.h"
#include "sdr_acc.h"
#if defined(WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#include <unistd.h>
#include <sched.h>
#endif

#define ACC_YIELD_BINS 22   // Doppler bins to release CPU (as sdr_search_code_cpu())

// global variables ------------------------------------------------------------
static int acc_type = SDR_ACC_CPU;         // backend type
static char acc_name[256] = "cpu";         // backend name
static const sdr_acc_plugin_t *acc_plugin = NULL; // backend plugin
static void *acc_lib = NULL;               // backend plugin library handle
static sdr_acc_stat_t acc_stat = {0};      // backend statistics
static sdr_mutex_t acc_mtx = SDR_MUTEX_INIT; // lock for statistics
static int trk_type = SDR_ACC_CPU;         // tracking backend type
static char trk_name[64] = "cpu";          // tracking backend name
static const sdr_acc_plugin_t *trk_plugin = NULL; // tracking backend plugin
static void *trk_lib = NULL;               // tracking backend plugin library
static sdr_acc_stat_t trk_stat = {0};      // tracking backend statistics

// time (s) --------------------------------------------------------------------
static double acc_time(void)
{
#if defined(WIN32)
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart / f.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
#endif
}

// open plugin library ---------------------------------------------------------
static void *lib_open(const char *path)
{
#if defined(WIN32)
    return (void *)LoadLibraryA(path);
#else
    return dlopen(path, RTLD_NOW | RTLD_LOCAL);
#endif
}

// plugin library error message ----------------------------------------------
static const char *lib_err(void)
{
#if defined(WIN32)
    return "";
#else
    const char *err = dlerror();
    return err ? err : "";
#endif
}

// get plugin library symbol ---------------------------------------------------
static void *lib_sym(void *lib, const char *name)
{
#if defined(WIN32)
    return (void *)GetProcAddress((HMODULE)lib, name);
#else
    return dlsym(lib, name);
#endif
}

// close plugin library --------------------------------------------------------
static void lib_close(void *lib)
{
#if defined(WIN32)
    FreeLibrary((HMODULE)lib);
#else
    dlclose(lib);
#endif
}

// directory of executable -----------------------------------------------------
static int exe_dir(char *dir, int size)
{
    int n = 0;
#if defined(WIN32)
    n = (int)GetModuleFileNameA(NULL, dir, size);
#elif defined(__linux__)
    n = (int)readlink("/proc/self/exe", dir, size - 1);
#endif
    if (n <= 0 || n >= size) return 0;
    dir[n] = '\0';
    char *p = strrchr(dir, '/');
    if (!p) p = strrchr(dir, '\\');
    if (!p) return 0;
    *p = '\0';
    return 1;
}

// load backend plugin (SDR_ACC_PATH: directory of plugin libraries) ----------
static int load_plugin(const char *name, const char *opt, void **lib_p,
    const sdr_acc_plugin_t **plugin_p)
{
#if defined(WIN32)
    const char *ext = ".dll", *ldir = "win32";
#elif defined(MACOS)
    const char *ext = ".so", *ldir = "macos";
#else
    const char *ext = ".so", *ldir = "linux";
#endif
    char file[128], path[1024], dir[512];
    const char *env = getenv("SDR_ACC_PATH");
    void *lib = NULL;

    snprintf(file, sizeof(file), "libsdr_acc_%s%s", name, ext);
    if (env && *env) {
        snprintf(path, sizeof(path), "%.512s/%s", env, file);
        lib = lib_open(path);
    }
    if (!lib && exe_dir(dir, sizeof(dir))) {
        snprintf(path, sizeof(path), "%s/%s", dir, file);
        if (!(lib = lib_open(path))) {
            snprintf(path, sizeof(path), "%s/../lib/%s/%s", dir, ldir, file);
            lib = lib_open(path);
        }
    }
    if (!lib && !(lib = lib_open(file))) {
        fprintf(stderr, "accelerator plugin load error: %s (%s)\n", file,
            lib_err());
        return 0;
    }
    sdr_acc_plugin_func_t func =
        (sdr_acc_plugin_func_t)lib_sym(lib, "sdr_acc_plugin");
    const sdr_acc_plugin_t *plugin = func ? func() : NULL;
    if (!plugin || plugin->api_ver < SDR_ACC_API_MIN ||
        plugin->api_ver > SDR_ACC_API_VER) {
        fprintf(stderr, "accelerator plugin version error: %s\n", file);
        lib_close(lib);
        return 0;
    }
    if (!plugin->init(opt)) {
        fprintf(stderr, "accelerator plugin init error: %s %s\n", file, opt);
        lib_close(lib);
        return 0;
    }
    *lib_p = lib;
    *plugin_p = plugin;
    return 1;
}

//------------------------------------------------------------------------------
//  Initialize accelerator backend. Without the call, the CPU reference
//  backend is used.
//
//  args:
//      backend  (I)  backend ("cpu", "cpu_opt", "npu", "<plugin>[:<opt>]")
//
//  return:
//      Status (1: OK, 0: error (CPU reference backend used))
//
int sdr_acc_init(const char *backend)
{
    char name[256], *opt;

    sdr_acc_exit();
    if (!strcmp(backend, "cpu")) {
        return 1;
    }
    if (!strcmp(backend, "cpu_opt")) {
        acc_type = SDR_ACC_CPUOPT;
        strcpy(acc_name, backend);
        return 1;
    }
    snprintf(name, sizeof(name), "%s", !strcmp(backend, "npu") ? "ov:NPU" :
        backend);
    if ((opt = strchr(name, ':'))) *opt++ = '\0'; else opt = name + strlen(name);
    if (!load_plugin(name, opt, &acc_lib, &acc_plugin)) {
        return 0;
    }
    acc_type = SDR_ACC_PLUGIN;
    snprintf(acc_name, sizeof(acc_name), "%s", backend);
    return 1;
}

//------------------------------------------------------------------------------
//  Finalize accelerator backend and restore the CPU reference backend.
//
void sdr_acc_exit(void)
{
    if (acc_plugin) {
        acc_plugin->exit();
        acc_plugin = NULL;
    }
    if (acc_lib) {
        lib_close(acc_lib);
        acc_lib = NULL;
    }
    acc_type = SDR_ACC_CPU;
    strcpy(acc_name, "cpu");
}

// get accelerator backend type and name ---------------------------------------
int sdr_acc_type(const char **name)
{
    if (name) *name = acc_name;
    return acc_type;
}

// get (and reset) accelerator backend statistics ------------------------------
void sdr_acc_stat(sdr_acc_stat_t *stat, int reset)
{
    sdr_mutex_lock(&acc_mtx);
    if (stat) *stat = acc_stat;
    if (reset) memset(&acc_stat, 0, sizeof(acc_stat));
    sdr_mutex_unlock(&acc_mtx);
}

// update accelerator backend statistics ---------------------------------------
static void update_stat(int stat, double t, double t_conv,
    const sdr_acc_prof_t *prof)
{
    sdr_mutex_lock(&acc_mtx);
    if (stat) {
        acc_stat.ncall++;
        acc_stat.t_total += t;
        acc_stat.t_conv += t_conv;
        acc_stat.t_in += prof->t_in;
        acc_stat.t_exec += prof->t_exec;
        acc_stat.t_out += prof->t_out;
        acc_stat.bytes_in += prof->bytes_in;
        acc_stat.bytes_out += prof->bytes_out;
    } else {
        acc_stat.nfall++;
    }
    sdr_mutex_unlock(&acc_mtx);
}

// persistent thread-local buffer ----------------------------------------------
static void *acc_buff(int idx, size_t size)
{
    static __thread void *buff[4] = {NULL};
    static __thread size_t size_buff[4] = {0};

    if (size > size_buff[idx]) {
        sdr_cpx_free((sdr_cpx_t *)buff[idx]);
        buff[idx] = sdr_cpx_malloc((int)((size + 7) / 8));
        size_buff[idx] = size;
    }
    return buff[idx];
}

// IF data to complex float tensor ---------------------------------------------
static void buff_to_tensor(const sdr_buff_t *buff, int ix, int N, float *x)
{
    float LUT[256][2];
    for (int i = 0; i < 256; i++) {
        LUT[i][0] = SDR_CPX8_I(i);
        LUT[i][1] = SDR_CPX8_Q(i);
    }
    for (int i = 0, j = ix; i < N; i++, j++) {
        if (j >= buff->N) j = 0;
        x[i*2  ] = LUT[buff->data[j]][0];
        x[i*2+1] = LUT[buff->data[j]][1];
    }
}

// IF data to complex int8 tensor ----------------------------------------------
static void buff_to_i8(const sdr_buff_t *buff, int ix, int N, int8_t *x)
{
    for (int i = 0, j = ix; i < N; i++, j++) {
        if (j >= buff->N) j = 0;
        x[i*2  ] = SDR_CPX8_I(buff->data[j]);
        x[i*2+1] = SDR_CPX8_Q(buff->data[j]);
    }
}

// parallel code search by plugin ----------------------------------------------
static int search_code_plugin(const sdr_cpx_t *code_fft, const sdr_buff_t *buff,
    int ix, int N, int Nmax, double fs, double fi, const float *fds,
    int len_fds, float *P)
{
    sdr_acc_prof_t prof = {0};
    double t0 = acc_time(), t1;
    double *fc = (double *)acc_buff(1, sizeof(double) * len_fds);
    int stat;
    
    for (int i = 0; i < len_fds; i++) {
        fc[i] = fi + fds[i];
    }
    if (acc_plugin->search_code_i8) { // int8 IF samples
        int8_t *x8 = (int8_t *)acc_buff(0, 2 * (size_t)N);
        buff_to_i8(buff, ix, N, x8);
        t1 = acc_time();
        stat = acc_plugin->search_code_i8(x8, (const float *)code_fft, N, fs,
            fc, len_fds, Nmax, P, N, &prof);
    }
    else {
        float *x = (float *)acc_buff(0, sizeof(float) * 2 * N);
        buff_to_tensor(buff, ix, N, x);
        t1 = acc_time();
        stat = acc_plugin->search_code(x, (const float *)code_fft, N, fs, fc,
            len_fds, Nmax, P, N, &prof);
    }
    update_stat(stat, acc_time() - t0, t1 - t0, &prof);
    return stat;
}

// parallel code search by CPU optimized (stat: update statistics) -------------
static int search_code_opt(const sdr_cpx_t *code_fft, const sdr_buff_t *buff,
    int ix, int N, int Nmax, double fs, double fi, const float *fds,
    int len_fds, float *P, int stat)
{
    sdr_acc_prof_t prof = {0};
    double t0 = acc_time();

    // Doppler bin shifts (DFT bins): exp(-j*2*pi*k*n/N) <-> X[m+k]
    int *sft = (int *)acc_buff(2, sizeof(int) * len_fds);
    for (int i = 0; i < len_fds; i++) {
        double s = (fds[i] - fds[0]) * N / fs;
        if (fabs(s - floor(s + 0.5)) > 1e-6) {
            if (stat) update_stat(0, 0.0, 0.0, &prof);
            return 0;
        }
        sft[i] = (((int)floor(s + 0.5)) % N + N) % N;
    }
    sdr_cpx_t *X = (sdr_cpx_t *)acc_buff(0, sizeof(sdr_cpx_t) * 2 * N);
    sdr_cpx_t *C = (sdr_cpx_t *)acc_buff(1, sizeof(sdr_cpx_t) * 2 * N);
    sdr_cpx16_t *IQ = (sdr_cpx16_t *)C;

    // mix carrier of the first Doppler bin and DFT (as sdr_corr_fft())
    sdr_mix_carr(buff, ix, N, fs, fi + fds[0], 0.0, IQ);
    for (int i = N - 1; i >= 0; i--) {
        X[N+i][0] = IQ[i].I * SDR_CSCALE;
        X[N+i][1] = IQ[i].Q * SDR_CSCALE;
    }
    if (!sdr_cpx_fft(X + N, N, SDR_FFT_FORWARD, X)) {
        if (stat) update_stat(0, 0.0, 0.0, &prof);
        return 0;
    }
    for (int i = 0; i < len_fds; i++) {
        int s = sft[i];

        // shift DFT of IF data and multiply code DFT
        sdr_cpx_mul(X + s, code_fft, N - s, 1.0f / N / N, C + N);
        sdr_cpx_mul(X, code_fft + N - s, s, 1.0f / N / N, C + N + N - s);
        (void)sdr_cpx_fft(C + N, N, SDR_FFT_BACKWARD, C);

        // add correlation power
        for (int j = 0; j < Nmax; j++) {
            P[i*N+j] += C[j][0] * C[j][0] + C[j][1] * C[j][1];
        }
        if (i % ACC_YIELD_BINS == ACC_YIELD_BINS - 1) { // release cpu
            sdr_sleep_msec(1);
        }
    }
    double t = acc_time() - t0;
    prof.t_exec = t;
    if (stat) update_stat(1, t, 0.0, &prof);
    return 1;
}

//------------------------------------------------------------------------------
//  Parallel code search by accelerator backend. The arguments are same as
//  sdr_search_code2(). The results are equivalent to the CPU reference within
//  the numerical tolerance of the backend. If a plugin does not support the
//  shape, the CPU optimized backend is used.
//
//  return:
//      Status (1: OK, 0: CPU reference backend or not supported by backend)
//
int sdr_acc_search_code(const sdr_cpx_t *code_fft, double T,
    const sdr_buff_t *buff, int ix, int N, int Nmax, double fs, double fi,
    const float *fds, int len_fds, float *P)
{
    if (Nmax <= 0 || Nmax > N) {
        return 0;
    }
    if (acc_type == SDR_ACC_CPUOPT) {
        return search_code_opt(code_fft, buff, ix, N, Nmax, fs, fi, fds,
            len_fds, P, 1);
    }
    if (acc_type == SDR_ACC_PLUGIN) { // CPU optimized if not supported
        // (counted as a fallback of the plugin)
        return search_code_plugin(code_fft, buff, ix, N, Nmax, fs, fi, fds,
            len_fds, P) || search_code_opt(code_fft, buff, ix, N, Nmax, fs, fi,
            fds, len_fds, P, 0);
    }
    return 0;
}

//------------------------------------------------------------------------------
//  Check whether the accelerator backend supports the parallel code search of
//  multiple windows (sdr_acc_search_code_multi()).
//
//  return:
//      Status (1: supported, 0: not supported)
//
int sdr_acc_search_multi(void)
{
    return acc_type == SDR_ACC_PLUGIN && acc_plugin->api_ver >= 4 &&
        acc_plugin->search_code_i8_multi;
}

//------------------------------------------------------------------------------
//  Parallel code search of multiple windows by accelerator backend. The
//  correlation powers of the windows starting at buffer indices ix[NW] are
//  added to P. Other arguments are same as sdr_acc_search_code().
//
//  return:
//      Status (1: OK, 0: not supported by backend (P not updated))
//
int sdr_acc_search_code_multi(const sdr_cpx_t *code_fft, double T,
    const sdr_buff_t *buff, const int *ix, int NW, int N, int Nmax, double fs,
    double fi, const float *fds, int len_fds, float *P)
{
    if (!sdr_acc_search_multi() || NW <= 0 || Nmax <= 0 || Nmax > N) {
        return 0;
    }
    sdr_acc_prof_t prof = {0};
    double t0 = acc_time(), t1;
    double *fc = (double *)acc_buff(1, sizeof(double) * len_fds);
    int8_t *x8 = (int8_t *)acc_buff(0, 2 * (size_t)N * NW);
    const int8_t **xs = (const int8_t **)acc_buff(3, sizeof(int8_t *) * NW);
    
    for (int i = 0; i < len_fds; i++) {
        fc[i] = fi + fds[i];
    }
    for (int i = 0; i < NW; i++) {
        buff_to_i8(buff, ix[i], N, x8 + 2 * (size_t)N * i);
        xs[i] = x8 + 2 * (size_t)N * i;
    }
    t1 = acc_time();
    int stat = acc_plugin->search_code_i8_multi(xs, NW, (const float *)code_fft,
        N, fs, fc, len_fds, Nmax, P, N, &prof);
    update_stat(stat, acc_time() - t0, t1 - t0, &prof);
    return stat;
}

//------------------------------------------------------------------------------
//  Tracking correlator backends
//------------------------------------------------------------------------------

#define MAX_TRK_TH 64       // max number of threads of cpu_mt backend

#if defined(WIN32)
typedef CONDITION_VARIABLE trk_cond_t;
#define cond_init(c)        InitializeConditionVariable(c)
#define cond_wait(c, m)     SleepConditionVariableSRW(c, m, INFINITE, 0)
#define cond_bcast(c)       WakeAllConditionVariable(c)
#else
typedef pthread_cond_t trk_cond_t;
#define cond_init(c)        pthread_cond_init(c, NULL)
#define cond_wait(c, m)     pthread_cond_wait(c, m)
#define cond_bcast(c)       pthread_cond_broadcast(c)
#endif

typedef struct {            // thread pool of cpu_mt backend
    int nth;                // number of worker threads
    int state;              // state (0: stop, 1: run)
    sdr_thread_t th[MAX_TRK_TH]; // worker threads
    sdr_mutex_t mtx;        // lock
    trk_cond_t cond_start, cond_done; // conditions of batch start and done
    sdr_corr_job_t *jobs;   // jobs of current batch
    int njob, next, ndone;  // number of jobs, next job, number of done jobs
    uint64_t gen;           // batch generation
} trk_pool_t;

static trk_pool_t trk_pool = {0};

// run standard correlator of job by CPU (reference) ---------------------------
static void corr_job_cpu(sdr_corr_job_t *job)
{
    sdr_corr_std(job->buff, job->ix, job->N, job->fs, job->fc, job->phi,
        job->code, job->code_sum, NULL, NULL, job->scale, job->coff, job->pos,
        job->npos, 0, job->corr, job->C);
}

// take and run jobs of current batch (call with pool locked) -----------------
static void pool_run(trk_pool_t *pool)
{
    while (pool->next < pool->njob) {
        sdr_corr_job_t *job = pool->jobs + pool->next++;
        sdr_mutex_unlock(&pool->mtx);
        corr_job_cpu(job);
        sdr_mutex_lock(&pool->mtx);
        if (++pool->ndone == pool->njob) {
            cond_bcast(&pool->cond_done);
        }
    }
}

// worker thread of thread pool ------------------------------------------------
static void *pool_thread(void *arg)
{
    trk_pool_t *pool = (trk_pool_t *)arg;
    uint64_t gen = 0;
    
    sdr_mutex_lock(&pool->mtx);
    while (pool->state) {
        if (pool->gen == gen) {
            cond_wait(&pool->cond_start, &pool->mtx);
            continue;
        }
        gen = pool->gen;
        pool_run(pool);
    }
    sdr_mutex_unlock(&pool->mtx);
    return NULL;
}

// start thread pool -----------------------------------------------------------
static int pool_start(trk_pool_t *pool, int nth)
{
    memset(pool, 0, sizeof(trk_pool_t));
    sdr_mutex_init(&pool->mtx);
    cond_init(&pool->cond_start);
    cond_init(&pool->cond_done);
    pool->state = 1;
    for (int i = 0; i < nth - 1 && i < MAX_TRK_TH; i++) { // + caller thread
        if (!sdr_thread_create(&pool->th[i], pool_thread, pool)) break;
        pool->nth++;
    }
    return 1;
}

// stop thread pool ------------------------------------------------------------
static void pool_stop(trk_pool_t *pool)
{
    sdr_mutex_lock(&pool->mtx);
    pool->state = 0;
    cond_bcast(&pool->cond_start);
    sdr_mutex_unlock(&pool->mtx);
    for (int i = 0; i < pool->nth; i++) {
        sdr_thread_join(pool->th[i]);
    }
    pool->nth = 0;
}

// run jobs by thread pool -----------------------------------------------------
static void pool_corr_batch(trk_pool_t *pool, sdr_corr_job_t *jobs, int njob)
{
    sdr_mutex_lock(&pool->mtx);
    pool->jobs = jobs;
    pool->njob = njob;
    pool->next = pool->ndone = 0;
    pool->gen++;
    cond_bcast(&pool->cond_start);
    pool_run(pool); // caller thread also runs jobs
    while (pool->ndone < pool->njob) {
        cond_wait(&pool->cond_done, &pool->mtx);
    }
    sdr_mutex_unlock(&pool->mtx);
}

// number of CPU cores ---------------------------------------------------------
static int num_cores(void)
{
#if defined(WIN32)
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    return (int)si.dwNumberOfProcessors;
#else
#if defined(__linux__)
    cpu_set_t set;
    if (sched_getaffinity(0, sizeof(set), &set) == 0) return CPU_COUNT(&set);
#endif
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    return n > 0 ? (int)n : 1;
#endif
}

// update tracking backend statistics ------------------------------------------
static void update_trk_stat(int stat, int njob, double t, double t_conv,
    const sdr_acc_prof_t *prof)
{
    sdr_mutex_lock(&acc_mtx);
    if (stat) {
        trk_stat.ncall++;
        trk_stat.njob += njob;
        trk_stat.t_total += t;
        trk_stat.t_conv += t_conv;
        if (prof) {
            trk_stat.t_in += prof->t_in;
            trk_stat.t_exec += prof->t_exec;
            trk_stat.t_out += prof->t_out;
            trk_stat.bytes_in += prof->bytes_in;
            trk_stat.bytes_out += prof->bytes_out;
        }
    } else {
        trk_stat.nfall++;
    }
    sdr_mutex_unlock(&acc_mtx);
}

// growable thread-local buffer of tracking ------------------------------------
static void *trk_buff(int idx, size_t size)
{
    static __thread void *buff[5] = {NULL};
    static __thread size_t size_buff[5] = {0};
    
    if (size > size_buff[idx]) {
        sdr_free(buff[idx]);
        buff[idx] = sdr_malloc(size);
        size_buff[idx] = size;
    }
    return buff[idx];
}

// batched correlators by plugin (jobs sharing the same IF samples) ------------
static int corr_batch_plugin(sdr_corr_job_t *jobs, int njob)
{
    const sdr_buff_t *buff = jobs[0].buff;
    int N = jobs[0].N, ix = jobs[0].ix, ntap = 0;
    sdr_acc_prof_t prof = {0};
    double t0 = acc_time();
    
    for (int i = 0; i < njob; i++) {
        ntap += jobs[i].npos;
    }
    int8_t *x = (int8_t *)trk_buff(0, 2 * N);
    sdr_acc_corr_t *corr = (sdr_acc_corr_t *)trk_buff(1,
        sizeof(sdr_acc_corr_t) * njob);
    const int8_t **code = (const int8_t **)trk_buff(2, sizeof(int8_t *) * ntap);
    int *js = (int *)trk_buff(3, sizeof(int) * ntap);
    float *out = (float *)trk_buff(4, sizeof(float) * 4 * ntap);
    
    // IF samples (shared by jobs)
    for (int i = 0, j = ix; i < N; i++, j++) {
        if (j >= buff->N) j = 0;
        x[i*2  ] = SDR_CPX8_I(buff->data[j]);
        x[i*2+1] = SDR_CPX8_Q(buff->data[j]);
    }
    // code slices and wrap-around samples of correlators (as sdr_corr_std2())
    for (int i = 0, m = 0; i < njob; i++) {
        sdr_corr_job_t *job = jobs + i;
        corr[i].fc = job->fc;
        corr[i].phi = job->phi;
        corr[i].n = job->npos;
        corr[i].scale = job->scale;
        corr[i].code = code + m;
        corr[i].j = js + m;
        corr[i].corr = out + 4 * m;
        for (int k = 0; k < job->npos; k++, m++) {
            double p = job->coff + job->pos[k];
            int j = (int)floor(p), b = (int)((p - j) * SDR_N_CODES);
            if (j < 0) j += N; else if (j >= N) j -= N;
            js[m] = j;
            code[m] = job->code + b * N;
        }
    }
    double t1 = acc_time();
    if (!trk_plugin->corr_batch(x, N, jobs[0].fs, njob, corr, &prof)) {
        return 0;
    }
    // combine correlations before/after wrap-around (as sdr_corr_std2())
    for (int i = 0; i < njob; i++) {
        sdr_corr_job_t *job = jobs + i;
        const float *c = corr[i].corr;
        float sc = 1.0f / job->scale, dot_EPL = 0.0f;
        for (int k = 0; k < 3 && k < job->npos; k++) {
            dot_EPL += c[k*4] * c[k*4+2] + c[k*4+1] * c[k*4+3];
        }
        float sign = dot_EPL < 0.0f ? -1.0f : 1.0f;
        for (int k = 0; k < job->npos; k++) {
            job->corr[k][0] = (c[k*4  ] * sc + sign * c[k*4+2] * sc) / N;
            job->corr[k][1] = (c[k*4+1] * sc + sign * c[k*4+3] * sc) / N;
        }
        job->C[0][0] = c[0] * sc;
        job->C[0][1] = c[1] * sc;
        job->C[1][0] = c[2] * sc;
        job->C[1][1] = c[3] * sc;
    }
    update_trk_stat(1, njob, acc_time() - t0, t1 - t0, &prof);
    return 1;
}

//------------------------------------------------------------------------------
//  Initialize tracking correlator backend. Without the call, the CPU reference
//  backend is used.
//
//  args:
//      backend  (I)  backend ("cpu", "cpu_mt[:n]", "npu", "<plugin>[:<opt>]")
//
//  return:
//      Status (1: OK, 0: error (CPU reference backend used))
//
int sdr_acc_trk_init(const char *backend)
{
    char name[64], *opt;
    
    sdr_acc_trk_exit();
    if (!strcmp(backend, "cpu")) {
        return 1;
    }
    if (!strncmp(backend, "cpu_mt", 6)) {
        int nth = backend[6] == ':' ? atoi(backend + 7) : num_cores();
        if (nth < 1) nth = 1;
        pool_start(&trk_pool, nth);
        trk_type = SDR_ACC_CPUOPT;
        snprintf(trk_name, sizeof(trk_name), "cpu_mt:%d", trk_pool.nth + 1);
        return 1;
    }
    snprintf(name, sizeof(name), "%s", !strcmp(backend, "npu") ? "ov:NPU" :
        backend);
    if ((opt = strchr(name, ':'))) *opt++ = '\0'; else opt = name + strlen(name);
    if (!load_plugin(name, opt, &trk_lib, &trk_plugin)) {
        return 0;
    }
    if (!trk_plugin->corr_batch) {
        fprintf(stderr, "accelerator plugin: no tracking correlator %s\n",
            name);
        sdr_acc_trk_exit();
        return 0;
    }
    trk_type = SDR_ACC_PLUGIN;
    snprintf(trk_name, sizeof(trk_name), "%s", backend);
    return 1;
}

//------------------------------------------------------------------------------
//  Finalize tracking correlator backend and restore the CPU reference backend.
//
void sdr_acc_trk_exit(void)
{
    if (trk_type == SDR_ACC_CPUOPT) {
        pool_stop(&trk_pool);
    }
    if (trk_plugin) {
        trk_plugin->exit();
        trk_plugin = NULL;
    }
    if (trk_lib) {
        lib_close(trk_lib);
        trk_lib = NULL;
    }
    trk_type = SDR_ACC_CPU;
    strcpy(trk_name, "cpu");
}

// get tracking correlator backend type and name -------------------------------
int sdr_acc_trk_type(const char **name)
{
    if (name) *name = trk_name;
    return trk_type;
}

// get (and reset) tracking correlator backend statistics ----------------------
void sdr_acc_trk_stat(sdr_acc_stat_t *stat, int reset)
{
    sdr_mutex_lock(&acc_mtx);
    if (stat) *stat = trk_stat;
    if (reset) memset(&trk_stat, 0, sizeof(trk_stat));
    sdr_mutex_unlock(&acc_mtx);
}

//------------------------------------------------------------------------------
//  Batched tracking correlators. The standard correlators (sdr_corr_std()) of
//  the jobs set by sdr_ch_update_pre() are executed by the tracking correlator
//  backend. The jobs of a batch should share the same IF samples (IF data
//  buffer, index and length) for the plugin backends, e.g. the channels of an
//  RF channel with the same code period. The results of the CPU backends are
//  identical to sdr_corr_std().
//
//  args:
//      jobs     (IO) correlator jobs (job->corr and job->C are output)
//      njob     (I)  number of jobs
//
//  return:
//      none
//
void sdr_acc_corr_batch(sdr_corr_job_t *jobs, int njob)
{
    if (njob <= 0) return;
    
    if (trk_type == SDR_ACC_CPUOPT) {
        double t0 = acc_time();
        pool_corr_batch(&trk_pool, jobs, njob);
        update_trk_stat(1, njob, acc_time() - t0, 0.0, NULL);
        return;
    }
    if (trk_type == SDR_ACC_PLUGIN) {
        // split into runs of jobs sharing the same IF samples
        for (int i = 0, j; i < njob; i = j) {
            for (j = i + 1; j < njob && jobs[j].buff == jobs[i].buff &&
                jobs[j].ix == jobs[i].ix && jobs[j].N == jobs[i].N &&
                jobs[j].fs == jobs[i].fs; j++) ;
            if (corr_batch_plugin(jobs + i, j - i)) continue;
            update_trk_stat(0, 0, 0.0, 0.0, NULL);
            for (int k = i; k < j; k++) corr_job_cpu(jobs + k); // fallback
        }
        return;
    }
    double t0 = acc_time();
    for (int i = 0; i < njob; i++) {
        corr_job_cpu(jobs + i);
    }
    update_trk_stat(1, njob, acc_time() - t0, 0.0, NULL);
}
