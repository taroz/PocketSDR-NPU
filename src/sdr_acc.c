//
//  Pocket SDR C Library - GNSS SDR Acquisition Accelerator Functions.
//
//  Interface between the parallel code search of the receiver and an
//  accelerator backend (NPU etc.) loaded as a plugin (sdr_acc.h).
//
//    backend   description
//    cpu       CPU (sdr_search_code()) (default)
//    <p>[:opt] plugin libsdr_acc_<p>.so (.dll) with option string opt
//
//  Author:
//  T.SUZUKI
//
//  History:
//  2026-10-16  1.0  new
//
#include "pocket_sdr.h"
#include "sdr_acc.h"
#if defined(WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#include <unistd.h>
#endif

// global variables ------------------------------------------------------------
static int acc_type = SDR_ACC_CPU;         // backend type
static char acc_name[256] = "cpu";         // backend name
static const sdr_acc_plugin_t *acc_plugin = NULL; // backend plugin
static void *acc_lib = NULL;               // backend plugin library handle
static sdr_acc_stat_t acc_stat = {0};      // backend statistics
static sdr_mutex_t acc_mtx = SDR_MUTEX_INIT; // lock for statistics

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
    if (!plugin || plugin->api_ver != SDR_ACC_API_VER) {
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
//  Initialize accelerator backend. Without the call, the CPU is used.
//
//  args:
//      backend  (I)  backend ("cpu", "<plugin>[:<opt>]")
//
//  return:
//      Status (1: OK, 0: error (CPU used))
//
int sdr_acc_init(const char *backend)
{
    char name[256], *opt;

    sdr_acc_exit();
    if (!strcmp(backend, "cpu")) {
        return 1;
    }
    snprintf(name, sizeof(name), "%s", backend);
    if ((opt = strchr(name, ':'))) *opt++ = '\0'; else opt = name + strlen(name);
    if (!load_plugin(name, opt, &acc_lib, &acc_plugin)) {
        return 0;
    }
    acc_type = SDR_ACC_PLUGIN;
    snprintf(acc_name, sizeof(acc_name), "%s", backend);
    return 1;
}

//------------------------------------------------------------------------------
//  Finalize accelerator backend (the CPU is used).
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
    int8_t *x8 = (int8_t *)acc_buff(0, 2 * (size_t)N);
    
    for (int i = 0; i < len_fds; i++) {
        fc[i] = fi + fds[i];
    }
    buff_to_i8(buff, ix, N, x8);
    t1 = acc_time();
    int stat = acc_plugin->search_code(x8, (const float *)code_fft, N, fs, fc,
        len_fds, Nmax, P, N, &prof);
    update_stat(stat, acc_time() - t0, t1 - t0, &prof);
    return stat;
}

//------------------------------------------------------------------------------
//  Parallel code search by accelerator backend. The arguments are same as
//  sdr_search_code2(). The results are equivalent to sdr_search_code() within
//  the numerical tolerance of the backend.
//
//  return:
//      Status (1: OK, 0: CPU backend or not supported by backend (counted as a
//      fallback, P not updated))
//
int sdr_acc_search_code(const sdr_cpx_t *code_fft, double T,
    const sdr_buff_t *buff, int ix, int N, int Nmax, double fs, double fi,
    const float *fds, int len_fds, float *P)
{
    if (acc_type != SDR_ACC_PLUGIN || Nmax <= 0 || Nmax > N) {
        return 0;
    }
    return search_code_plugin(code_fft, buff, ix, N, Nmax, fs, fi, fds,
        len_fds, P);
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
    return acc_type == SDR_ACC_PLUGIN && acc_plugin->search_code_multi;
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
    int stat = acc_plugin->search_code_multi(xs, NW, (const float *)code_fft,
        N, fs, fc, len_fds, Nmax, P, N, &prof);
    update_stat(stat, acc_time() - t0, t1 - t0, &prof);
    return stat;
}
