#include <stdint.h>
#include <stdarg.h>
#include <time.h>

int32_t omp_get_max_threads(void){return 1;}
int32_t omp_get_num_threads(void){return 1;}
int32_t omp_get_thread_num(void){return 0;}
int32_t omp_in_parallel(void){return 0;}
int32_t omp_get_dynamic(void){return 0;}
void omp_set_dynamic(int32_t v){(void)v;}
void omp_set_num_threads(int32_t n){(void)n;}
void omp_set_nested(int32_t v){(void)v;}
int32_t omp_get_nested(void){return 0;}
int32_t omp_get_thread_limit(void){return 1;}
int32_t omp_get_num_procs(void){return 1;}
int32_t kmp_get_blocktime(void){return 0;}
void kmp_set_blocktime(int32_t v){(void)v;}
int32_t __kmpc_global_thread_num(void* l){(void)l;return 0;}
void __kmpc_push_num_threads(void* l,int32_t g,int32_t n){(void)l;(void)g;(void)n;}
void __kmpc_begin(void* l){(void)l;}
void __kmpc_end(void* l){(void)l;}
void __kmpc_serialized_parallel(void* l,int32_t g){(void)l;(void)g;}
void __kmpc_end_serialized_parallel(void* l,int32_t g){(void)l;(void)g;}
void __kmpc_barrier(void* l,int32_t g){(void)l;(void)g;}
void __kmpc_critical(void* l,int32_t g,void* c){(void)l;(void)g;(void)c;}
void __kmpc_endcritical(void* l,int32_t g,void* c){(void)l;(void)g;(void)c;}
void __kmpc_for_static_fini(void* l,int32_t g){(void)l;(void)g;}

typedef void (*microtask_t)(int32_t*,int32_t*,...);
void __kmpc_fork_call(void* l,int32_t argc,void* mt,...){
    (void)l;
    int32_t g=0,n=1;
    void* a[8]={0,0,0,0,0,0,0,0};
    va_list ap; va_start(ap,mt);
    for(int i=0;i<argc&&i<8;i++) a[i]=va_arg(ap,void*);
    va_end(ap);
    ((microtask_t)mt)(&g,&n,a[0],a[1],a[2],a[3],a[4],a[5],a[6],a[7]);
}

void __kmpc_for_static_init_4(void* l,int32_t g,int32_t s,int32_t* li,int32_t* lo,int32_t* up,int32_t* st,int32_t inc,int32_t ch){(void)l;(void)g;(void)s;(void)inc;(void)ch;if(li)*li=1;if(st)*st=(*up-*lo)+1;}
void __kmpc_for_static_init_4u(void* l,int32_t g,int32_t s,int32_t* li,uint32_t* lo,uint32_t* up,uint32_t* st,uint32_t inc,uint32_t ch){(void)l;(void)g;(void)s;(void)inc;(void)ch;if(li)*li=1;if(st)*st=(*up-*lo)+1;}
void __kmpc_for_static_init_8(void* l,int32_t g,int32_t s,int32_t* li,int64_t* lo,int64_t* up,int64_t* st,int64_t inc,int64_t ch){(void)l;(void)g;(void)s;(void)inc;(void)ch;if(li)*li=1;if(st)*st=(*up-*lo)+1;}
void __kmpc_for_static_init_8u(void* l,int32_t g,int32_t s,int32_t* li,uint64_t* lo,uint64_t* up,uint64_t* st,uint64_t inc,uint64_t ch){(void)l;(void)g;(void)s;(void)inc;(void)ch;if(li)*li=1;if(st)*st=(*up-*lo)+1;}
