#include <stdint.h>
#include <stdarg.h>
#include <time.h>
#include <pthread.h>

#define MAXT 8

static __thread int t_gtid = 0;
static __thread int t_ntid = 1;
static int g_team = 1;

typedef void (*microtask_t)(int32_t*, int32_t*, ...);

typedef struct {
    microtask_t mt;
    int argc;
    void* a[8];
    int nthreads;
} task_t;

static task_t g_task;
static pthread_t g_th[MAXT];
static int g_nworkers = 0;
static pthread_mutex_t g_mx = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t g_cv = PTHREAD_COND_INITIALIZER;
static pthread_cond_t g_done = PTHREAD_COND_INITIALIZER;
static volatile int g_state = 0;
static volatile int g_left = 0;

static pthread_mutex_t g_bmx = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t g_bcv = PTHREAD_COND_INITIALIZER;
static int g_bar_count = 0;
static int g_bar_gen = 0;

static pthread_mutex_t g_cmx = PTHREAD_MUTEX_INITIALIZER;

static void call_mt(const task_t* t, int g, int n) {
    int32_t gg = g, nn = n;
    switch (t->argc) {
    case 0: t->mt(&gg, &nn); break;
    case 1: t->mt(&gg, &nn, t->a[0]); break;
    case 2: t->mt(&gg, &nn, t->a[0], t->a[1]); break;
    case 3: t->mt(&gg, &nn, t->a[0], t->a[1], t->a[2]); break;
    case 4: t->mt(&gg, &nn, t->a[0], t->a[1], t->a[2], t->a[3]); break;
    case 5: t->mt(&gg, &nn, t->a[0], t->a[1], t->a[2], t->a[3], t->a[4]); break;
    case 6: t->mt(&gg, &nn, t->a[0], t->a[1], t->a[2], t->a[3], t->a[4], t->a[5]); break;
    case 7: t->mt(&gg, &nn, t->a[0], t->a[1], t->a[2], t->a[3], t->a[4], t->a[5], t->a[6]); break;
    default: t->mt(&gg, &nn, t->a[0], t->a[1], t->a[2], t->a[3], t->a[4], t->a[5], t->a[6], t->a[7]); break;
    }
}

static void* worker(void* arg) {
    int id = (int)(intptr_t)arg;
    for (;;) {
        pthread_mutex_lock(&g_mx);
        while (g_state == 0) pthread_cond_wait(&g_cv, &g_mx);
        if (g_state == 2) { pthread_mutex_unlock(&g_mx); return NULL; }
        int participate = (id < g_task.nthreads);
        pthread_mutex_unlock(&g_mx);
        if (participate) {
            t_gtid = id; t_ntid = g_task.nthreads;
            call_mt(&g_task, id, g_task.nthreads);
            pthread_mutex_lock(&g_mx);
            if (--g_left == 0) pthread_cond_signal(&g_done);
            pthread_mutex_unlock(&g_mx);
            t_ntid = 1; t_gtid = 0;
        }
    }
}

static void ensure_workers_locked(int n) {
    int need = n - 1;
    while (g_nworkers < need && g_nworkers < MAXT - 1) {
        pthread_create(&g_th[g_nworkers], NULL, worker, (void*)(intptr_t)(g_nworkers + 1));
        g_nworkers++;
    }
}

void __kmpc_fork_call(void* l, int argc, void* mt, ...) {
    (void)l;
    task_t t;
    t.mt = (microtask_t)mt;
    t.argc = argc > 8 ? 8 : argc;
    va_list ap; va_start(ap, mt);
    for (int i = 0; i < t.argc; i++) t.a[i] = va_arg(ap, void*);
    va_end(ap);
    int want = g_team;
    if (want > MAXT) want = MAXT;
    if (want < 1) want = 1;
    if (t_ntid != 1 || want == 1) {
        call_mt(&t, 0, 1);
        return;
    }
    t.nthreads = want;
    pthread_mutex_lock(&g_mx);
    ensure_workers_locked(want);
    g_task = t;
    g_left = want - 1;
    g_state = 1;
    pthread_cond_broadcast(&g_cv);
    pthread_mutex_unlock(&g_mx);
    t_gtid = 0; t_ntid = want;
    call_mt(&t, 0, want);
    pthread_mutex_lock(&g_mx);
    while (g_left > 0) pthread_cond_wait(&g_done, &g_mx);
    g_state = 0;
    pthread_mutex_unlock(&g_mx);
    t_ntid = 1; t_gtid = 0;
}

#define FOR_INIT(NAME, T) \
void NAME(void* l, int32_t g, int32_t s, int32_t* li, T* lo, T* up, T* st, T inc, T ch) { \
    (void)l; (void)g; (void)s; (void)inc; (void)ch; \
    int n = t_ntid; int id = t_gtid; \
    int64_t lo0 = (int64_t)*lo, hi0 = (int64_t)*up; \
    int64_t cnt = hi0 - lo0 + 1; \
    if (cnt <= 0) { if (li) *li = 0; *lo = (T)1; *up = (T)0; if (st) *st = (T)1; return; } \
    int64_t per = (cnt + n - 1) / n; \
    int64_t tlo = lo0 + (int64_t)id * per; \
    int64_t thi = tlo + per - 1; if (thi > hi0) thi = hi0; \
    if (tlo > hi0) { *lo = (T)1; *up = (T)0; if (li) *li = 0; } \
    else { *lo = (T)tlo; *up = (T)thi; if (li) *li = (thi == hi0); } \
    if (st) *st = (T)per; \
}
FOR_INIT(__kmpc_for_static_init_4, int32_t)
FOR_INIT(__kmpc_for_static_init_4u, uint32_t)
FOR_INIT(__kmpc_for_static_init_8, int64_t)
FOR_INIT(__kmpc_for_static_init_8u, uint64_t)

void __kmpc_for_static_fini(void* l, int32_t g){(void)l;(void)g;}
void __kmpc_begin(void* l){(void)l;}
void __kmpc_end(void* l){(void)l;}
void __kmpc_push_num_threads(void* l,int32_t g,int32_t n){(void)l;(void)g;(void)n;}
void __kmpc_serialized_parallel(void* l,int32_t g){(void)l;(void)g;}
void __kmpc_end_serialized_parallel(void* l,int32_t g){(void)l;(void)g;}
int32_t __kmpc_global_thread_num(void* l){(void)l;return t_gtid;}

void __kmpc_barrier(void* l, int32_t g) {
    (void)l; (void)g;
    pthread_mutex_lock(&g_bmx);
    int gen = g_bar_gen;
    if (++g_bar_count == t_ntid) { g_bar_count = 0; g_bar_gen++; pthread_cond_broadcast(&g_bcv); }
    else while (g_bar_gen == gen) pthread_cond_wait(&g_bcv, &g_bmx);
    pthread_mutex_unlock(&g_bmx);
}

void __kmpc_critical(void* l,int32_t g,void* c){(void)l;(void)g;(void)c;pthread_mutex_lock(&g_cmx);}
void __kmpc_endcritical(void* l,int32_t g,void* c){(void)l;(void)g;(void)c;pthread_mutex_unlock(&g_cmx);}

int32_t omp_get_max_threads(void){return g_team;}
int32_t omp_get_num_threads(void){return t_ntid;}
int32_t omp_get_thread_num(void){return t_gtid;}
int32_t omp_in_parallel(void){return t_ntid > 1;}
int32_t omp_get_dynamic(void){return 0;}
void omp_set_dynamic(int32_t v){(void)v;}
void omp_set_num_threads(int32_t n){ if(n<1)n=1; if(n>MAXT)n=MAXT; g_team=n; }
void omp_set_nested(int32_t v){(void)v;}
int32_t omp_get_nested(void){return 0;}
int32_t omp_get_thread_limit(void){return MAXT;}
int32_t omp_get_num_procs(void){return MAXT;}
void omp_set_schedule(int32_t a,int32_t b){(void)a;(void)b;}
int32_t omp_get_schedule(void){return 0;}
double omp_get_wtime(void){struct timespec ts;clock_gettime(CLOCK_MONOTONIC,&ts);return ts.tv_sec+ts.tv_nsec*1e-9;}
double omp_get_wtick(void){return 1e-9;}
int32_t kmp_get_blocktime(void){return 0;}
void kmp_set_blocktime(int32_t v){(void)v;}
