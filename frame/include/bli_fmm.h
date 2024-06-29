#ifndef BLISLAB_DGEMM_H
#define BLISLAB_DGEMM_H

// Allow C++ users to include this header file in their source code. However,
// we make the extern "C" conditional on whether we're using a C++ compiler,
// since regular C compilers don't understand the extern "C" construct.
#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include <immintrin.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include <omp.h>
#include "bli_plugin_fmm_blis.h"


// Determine the target operating system
#if defined(_WIN32) || defined(__CYGWIN__)
#define BL_OS_WINDOWS 1
#elif defined(__APPLE__) || defined(__MACH__)
#define BL_OS_OSX 1
#elif defined(__ANDROID__)
#define BL_OS_ANDROID 1
#elif defined(__linux__)
#define BL_OS_LINUX 1
#elif defined(__bgq__)
#define BL_OS_BGQ 1
#elif defined(__bg__)
#define BL_OS_BGP 1
#elif defined(__FreeBSD__) || defined(__NetBSD__) || defined(__OpenBSD__) || \
      defined(__bsdi__) || defined(__DragonFly__)
#define BL_OS_BSD 1
#else
#error "Cannot determine operating system"
#endif

// gettimeofday() needs this.
#if BL_OS_WINDOWS
  #include <time.h>
#elif BL_OS_OSX
  #include <mach/mach_time.h>
#else
  #include <sys/time.h>
  #include <time.h>
#endif

//#include "bl_config.h"

#define min( i, j ) ( (i)<(j) ? (i): (j) )

#define A( i, j )     A[ (j)*lda + (i) ]
#define B( i, j )     B[ (j)*ldb + (i) ]
#define C( i, j )     C[ (j)*ldc + (i) ]
#define C_ref( i, j ) C_ref[ (j)*ldc_ref + (i) ]
#define C( i, j )     C[ (j)*ldc + (i) ]

extern fmm_t STRASSEN_FMM;
extern fmm_t CLASSICAL_FMM;

#define DGEMM_MC 96
#define DGEMM_NC 4096 // 2052 //
#define DGEMM_KC 256
#define DGEMM_MR 8
#define DGEMM_NR 6

struct fmm_cntl_s
{
    cntl_t cntl; // cntl field must be present and come first.
    fmm_t* fmm;
    gemm_cntl_t* gemm_cntl; // to faciliate locating
};
typedef struct fmm_cntl_s fmm_cntl_t;


struct fmm_gemm_cntl_s
{
    fmm_cntl_t fmm_cntl;
    gemm_cntl_t gemm_cntl;
};
typedef struct fmm_gemm_cntl_s fmm_gemm_cntl_t;

struct fmm_gemm_cntl_alt_s
{
    gemm_cntl_t gemm_cntl;
    fmm_cntl_t fmm_cntl;
};
typedef struct fmm_gemm_cntl_alt_s fmm_gemm_cntl_alt_t;

void bli_fmm_gemm_cntl_init_pushb
     (
             ind_t        im,
             opid_t       family,
       const obj_t*       alpha,
             obj_t*       a,
             obj_t*       b,
       const obj_t*       beta,
             obj_t*       c,
       const cntx_t*      cntx,
             gemm_cntl_t* cntl,
             fmm_cntl_t* fmm_cntl,
             int variant
     );

void bli_fmm_gemm_cntl_init_var
     (
             ind_t        im,
             opid_t       family,
       const obj_t*       alpha,
             obj_t*       a,
             obj_t*       b,
       const obj_t*       beta,
             obj_t*       c,
       const cntx_t*      cntx,
             gemm_cntl_t* cntl,
             fmm_cntl_t* fmm_cntl,
             int variant
     );

void bli_fmm_gemm_cntl_init
     (
             ind_t        im,
             opid_t       family,
       const obj_t*       alpha,
             obj_t*       a,
             obj_t*       b,
       const obj_t*       beta,
             obj_t*       c,
       const cntx_t*      cntx,
             gemm_cntl_t* cntl,
             fmm_cntl_t* fmm_cntl
     );

void bli_l3_packb_fmm
     (
       const obj_t*  a,
       const obj_t*  b,
       const obj_t*  c,
       const cntx_t* cntx,
       const cntl_t* cntl,
             thrinfo_t* thread_par
     );

void do_fmm_test();

void bli_strassen_ab_ex_ex( 
        obj_t* alpha, 
        obj_t* A, 
        obj_t* B, 
        obj_t* beta, 
        obj_t* C,
        fmm_t* fmm,
        int variant
        );

void bli_strassen_ab_ex( 
        obj_t* alpha, 
        obj_t* A, 
        obj_t* B, 
        obj_t* beta, 
        obj_t* C,
        fmm_t* fmm
        );

void bli_strassen_ab( 
        obj_t* alpha, 
        obj_t* A, 
        obj_t* B, 
        obj_t* beta, 
        obj_t* C 
        );

void bli_strassen_ab_symm_ex( obj_t* alpha, obj_t* A, obj_t* B, obj_t* beta, obj_t* C, fmm_t* fmm);

void bli_strassen_ab_symm( obj_t* alpha, obj_t* A, obj_t* B, obj_t* beta, obj_t* C);

fmm_t new_fmm(const char* file_name);

fmm_t new_fmm_ex(const char* file_name, int nest_level);

fmm_t nest_fmm(fmm_t* fmm_a, fmm_t* fmm_b);

void print_fmm(fmm_t* fmm);

void free_fmm(fmm_t* fmm);

void test_bli_strassen( int m, int n, int k );

void bli_fmm_cntl
     (
       const obj_t*     a,
       const obj_t*     b,
       const obj_t*     c,
       const cntx_t*    cntx,
       const cntl_t*    cntl,
             thrinfo_t* thread_par
     );

void bl_dgemm(
        int    m,
        int    n,
        int    k,
        double *A,
        int    lda,
        double *B,
        int    ldb,
        double *C,
        int    ldc
        );

void bl_dgemm_beta0(
        int    m,
        int    n,
        int    k,
        double *A,
        int    lda,
        double *B,
        int    ldb,
        double *C,
        int    ldc
        );

void bl_dgemm_strassen(
        int    m,
        int    n,
        int    k,
        double *A,
        int    lda,
        double *B,
        int    ldb,
        double *C,
        int    ldc
        );

void bl_dgemm_strassen_abc(
        int    m,
        int    n,
        int    k,
        double *A,
        int    lda,
        double *B,
        int    ldb,
        double *C,
        int    ldc
        );

void bl_dgemm_strassen_ab(
        int    m,
        int    n,
        int    k,
        double *A,
        int    lda,
        double *B,
        int    ldb,
        double *C,
        int    ldc
        );

void bl_dgemm_strassen_naive(
        int    m,
        int    n,
        int    k,
        double *A,
        int    lda,
        double *B,
        int    ldb,
        double *C,
        int    ldc
        );

// XB = XA
void mkl_copym(
    int m,
    int n,
    double *XA,
    int lda,
    double *XB,
    int ldb
    );

// XB = XB + alpha * XA
void mkl_axpym(
        int m,
        int n,
        double *buf_alpha,
        double *XA,
        int lda,
        double *XB,
        int ldb
        );

double *bl_malloc_aligned(
        int    m,
        int    n,
        int    size
        );

void bl_free( void *p );

void bl_printmatrix(
        double *A,
        int    lda,
        int    m,
        int    n
        );

double bl_clock( void );
double bl_clock_helper();

void bl_dgemm_ref(
    int    m,
    int    n,
    int    k,
    double *XA,
    int    lda,
    double *XB,
    int    ldb,
    double *XC,
    int    ldc
    );

void bl_get_range( int n, int bf, int* start, int* end );

void bl_acquire_mpart( 
        int m,
        int n,
        double *src_buff,
        int lda,
        int x,
        int y,
        int i,
        int j,
        double **dst_buff
        );


double bl_compare_error( int ldc, int ldc_ref, int m, int n, double *C, double *C_ref );

void bl_dynamic_peeling( int m, int n, int k, double *A, int lda, double *B, int ldb, double *C, int ldc, int dim1, int dim2, int dim3 );

int bl_read_nway_from_env( char* env );

static double *glob_packA=NULL, *glob_packB=NULL;

void bl_finalize();

void bl_malloc_packing_pool( double **packA, double **packB, int n, int bl_ic_nt );

// End extern "C" construct block.
#ifdef __cplusplus
}
#endif

#endif
