//
//  Pocket SDR C library - SDR Common Functions.
//
//  Author:
//  T.TAKASU
//
//  History:
//  2021-10-03  0.1  new
//  2022-01-04  1.0  support Windows API.
//  2022-05-17  1.1  add API sdr_cpx_malloc(), sdr_cpx_free()
//  2022-07-08  1.2  add API sdr_get_time()
//                   move API sdr_cpx_malloc(), sdr_cpx_free() to sdr_func.c
//  2026-02-12  1.3  add API sdr_thread_create(), sdr_thread_join(),
//                   sdr_mutex_init(), sdr_mutex_lock(), sdr_mutex_unlock()
//  2026-10-16  1.4  add API sdr_thread_affinity()
//
#if !defined(WIN32) && !defined(_GNU_SOURCE)
#define _GNU_SOURCE         // pthread_setaffinity_np()
#endif
#include "pocket_sdr.h"
#ifdef WIN32
#include <timeapi.h>
#include <process.h>
#else
#include <time.h>
#include <sys/time.h>
#endif

// global variables ------------------------------------------------------------
static const char *lib_name = SDR_LIB_NAME;
static const char *lib_ver = SDR_LIB_VER;

//------------------------------------------------------------------------------
//  Allocate memory. memory is zero-cleared. If no memory allocated, it exits
//  the AP immediately with an error message.
//  
//  args:
//      size     (I)  memory size (bytes)
//
//  return:
//      memory pointer allocated.
//
void *sdr_malloc(size_t size)
{
    void *p;
    
    if (!(p = calloc(size, 1))) {
        fprintf(stderr, "memory allocation error size=%d\n", (int)size);
        exit(-1);
    }
    return p;
}

//------------------------------------------------------------------------------
//  Free memory allocated by sdr_malloc()
//  
//  args:
//      p        (I)  memory pointer allocated.
//
//  return:
//      none
//
void sdr_free(void *p)
{
    free(p);
}

//------------------------------------------------------------------------------
//  Get current time in UTC.
//  
//  args:
//      t        (O)  Current time as {year, month, day, hour, minute, second}
//
//  return:
//      none
//
void sdr_get_time(double *t)
{
#ifdef WIN32
    SYSTEMTIME ts;
    
    GetSystemTime(&ts); /* in UTC */
    t[0] = ts.wYear;
    t[1] = ts.wMonth;
    t[2] = ts.wDay;
    t[3] = ts.wHour;
    t[4] = ts.wMinute;
    t[5] = ts.wSecond + ts.wMilliseconds * 1E-3;
#else
    struct timeval tv;
    struct tm *tt;
    
    if (!gettimeofday(&tv, NULL) && (tt = gmtime(&tv.tv_sec))) {
        t[0] = tt->tm_year + 1900;
        t[1] = tt->tm_mon + 1;
        t[2] = tt->tm_mday;
        t[3] = tt->tm_hour;
        t[4] = tt->tm_min;
        t[5] = tt->tm_sec + tv.tv_usec * 1E-6;
    } else {
        t[0] = 1970.0;
        t[1] = t[2] = 1.0;
        t[3] = t[4] = t[5] = 0.0;
    }
#endif
}

//------------------------------------------------------------------------------
//  Get system tick (msec).
//  
//  args:
//      none
//
//  return:
//      system tick (ms)
//
uint32_t sdr_get_tick(void)
{
#ifdef WIN32
    return (uint32_t)timeGetTime();
#else
    struct timeval tv = {0, 0};
    
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000u + tv.tv_usec / 1000u;
#endif
}

//------------------------------------------------------------------------------
//  Sleep for milli-seconds.
//  
//  args:
//      msec     (I)  mill-seconds to sleep
//
//  return:
//      none
//
void sdr_sleep_msec(int msec)
{
    if (msec <= 0) return;
#ifdef WIN32
    Sleep(msec < 5 ? 1 : msec);
#else
    struct timespec ts = {0, 0};
    
    ts.tv_nsec = (long)(msec * 1000000);
    nanosleep(&ts, NULL);
#endif
}

//------------------------------------------------------------------------------
//  Get library name and version.
//  
//  args:
//      none
//
//  return:
//      library name or library version
//
const char *sdr_get_name(void)
{
    return lib_name;
}

const char *sdr_get_ver(void)
{
    return lib_ver;
}

//------------------------------------------------------------------------------
//  Create thread.
//
//  args:
//      thread   (O)  pointer to thread
//      func     (I)  thread function
//      arg      (I)  pointer to function argument
//
//  return:
//      status (1:OK,0:error)
//
#ifdef WIN32

typedef struct {void *(*func)(void *); void *arg;} _ctx_t;

static unsigned __stdcall _th_func(void *c)
{
    _ctx_t *ctx = (_ctx_t *)c;
    ctx->func(ctx->arg); // call thread function
    sdr_free(ctx);
    return 0;
}

int sdr_thread_create(sdr_thread_t *thread, void *(func)(void *), void *arg)
{
    uintptr_t h;
    unsigned tid;
    _ctx_t *ctx = (_ctx_t *)sdr_malloc(sizeof(_ctx_t));
    ctx->func = func;
    ctx->arg = arg;
    if (!(h = _beginthreadex(NULL, 0, _th_func, ctx, 0, &tid))) {
        sdr_free(ctx);
        return 0;
    }
    *thread = (HANDLE)h;
    return 1;
}

#else

int sdr_thread_create(sdr_thread_t *thread, void *(func)(void *), void *arg)
{
    return pthread_create(thread, NULL, func, arg) == 0;
}

#endif // WIN32

//------------------------------------------------------------------------------
//  Join thread. Blocked until the thread function exits.
//
//  args:
//      thread   (I)  thread
//
//  return:
//      none
//
void sdr_thread_join(sdr_thread_t thread)
{
#ifdef WIN32
    if (!thread) return;
    (void)WaitForSingleObject(thread, INFINITE);
    CloseHandle(thread);
#else
    (void)pthread_join(thread, NULL);
#endif
}

//------------------------------------------------------------------------------
//  Set CPU affinity of the calling thread. The threads created by the thread
//  after the call inherit the affinity.
//
//  args:
//      cpus     (I)  CPU list (e.g. "0-3", "4,6-7", SDR_CPU_ALL: all CPUs,
//                    "": no change)
//
//  return:
//      status (1:OK, 0:error or not supported)
//
int sdr_thread_affinity(const char *cpus)
{
    uint64_t mask = 0;
    const char *p = cpus;
    
    if (!cpus || !*cpus) return 1;
    while (*p) {
        char *q;
        long i = strtol(p, &q, 10), j = i;
        if (q == p || i < 0 || i > 63) return 0;
        if (*q == '-') {
            p = q + 1;
            j = strtol(p, &q, 10);
            if (q == p || j < i || j > 63) return 0;
        }
        for ( ; i <= j; i++) mask |= (uint64_t)1 << i;
        if (*q == ',') q++;
        else if (*q) return 0;
        p = q;
    }
#if defined(WIN32)
    DWORD_PTR pmask, smask;
    if (GetProcessAffinityMask(GetCurrentProcess(), &pmask, &smask)) {
        mask &= (uint64_t)pmask; // (CPUs not in the process ignored)
    }
    return mask && SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)mask);
#elif defined(__linux__)
    cpu_set_t set;
    CPU_ZERO(&set);
    for (int i = 0; i < 64; i++) {
        if (mask & ((uint64_t)1 << i)) CPU_SET(i, &set);
    }
    return pthread_setaffinity_np(pthread_self(), sizeof(set), &set) == 0;
#else
    return 0;
#endif
}

//------------------------------------------------------------------------------
//  Initialize mutex.
//
//  args:
//      mtx      (IO) mutex
//
//  return:
//      none
//
void sdr_mutex_init(sdr_mutex_t *mtx)
{
#ifdef WIN32
    if (mtx) InitializeSRWLock(mtx);
#else
    (void)pthread_mutex_init(mtx, NULL);
#endif
}

//------------------------------------------------------------------------------
//  Lock mutex.
//
//  args:
//      mtx      (IO) mutex
//
//  return:
//      none
//
void sdr_mutex_lock(sdr_mutex_t *mtx)
{
#ifdef WIN32
    if (mtx) AcquireSRWLockExclusive(mtx);
#else
    (void)pthread_mutex_lock(mtx);
#endif
}

//------------------------------------------------------------------------------
//  Unlock mutex.
//
//  args:
//      mtx      (IO) mutex
//
//  return:
//      none
//
void sdr_mutex_unlock(sdr_mutex_t *mtx)
{
#ifdef WIN32
    if (mtx) ReleaseSRWLockExclusive(mtx);
#else
    (void)pthread_mutex_unlock(mtx);
#endif
}
