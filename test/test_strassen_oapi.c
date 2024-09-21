// zeroing out packing buffer

#include "blis.h"
#include "bli_fmm.h"
#include <time.h>

#define X 15
#define N_CONST X
#define M_CONST X
#define K_CONST X

#define DEBUG_test 1
#define RUN_trials 0
#define SQUARE 0

#define _U( i,j ) fmm.U[ (i)*fmm.R + (j) ]
#define _V( i,j ) fmm.V[ (i)*fmm.R + (j) ]
#define _W( i,j ) fmm.W[ (i)*fmm.R + (j) ]

void my_mm(obj_t* A, obj_t* B, obj_t* C, dim_t m, dim_t n, dim_t k) {

    void*  buf_A    = bli_obj_buffer_at_off( A ); 
    inc_t  rs_A     = bli_obj_row_stride( A ); 
    inc_t  cs_A     = bli_obj_col_stride( A ); 
    double *buf_Aptr = buf_A;

    void*  buf_B    = bli_obj_buffer_at_off( B ); 
    inc_t  rs_B     = bli_obj_row_stride( B ); 
    inc_t  cs_B    = bli_obj_col_stride( B ); 
    double *buf_Bptr = buf_B;

    void*  buf_C    = bli_obj_buffer_at_off( C ); 
    inc_t  rs_C     = bli_obj_row_stride( C ); 
    inc_t  cs_C    = bli_obj_col_stride( C ); 
    double *buf_Cptr = buf_C;

    for (dim_t i = 0; i < m; i++) {
        for (dim_t j = 0; j < n; j++) {
            for(dim_t p = 0; p < k; p++) {
                buf_Cptr[i*rs_C + j * cs_C] += buf_Aptr[i*rs_A + p*cs_A] * buf_Bptr[p*rs_B + j*cs_B];
            }
        }
    }
}

float my_max_diff_f(obj_t* A, obj_t* B, dim_t m, dim_t n) {

    void*  buf_A    = bli_obj_buffer_at_off( A ); 
    inc_t  rs_A     = bli_obj_row_stride( A ); 
    inc_t  cs_A     = bli_obj_col_stride( A ); 
    float *buf_Aptr = buf_A;

    void*  buf_B    = bli_obj_buffer_at_off( B ); 
    inc_t  rs_B     = bli_obj_row_stride( B ); 
    inc_t  cs_B    = bli_obj_col_stride( B ); 
    float *buf_Bptr = buf_B;

    float max_diff = 0;
    for (dim_t i = 0; i < m; i++) {
        for (dim_t j = 0; j < n; j++) {
            float a = buf_Aptr[i*rs_A + j*cs_A];
            float b = buf_Bptr[i*rs_B + j*cs_B];
            if (a - b > max_diff) {
                max_diff = a - b;
            }
            else if (b - a > max_diff) {
                max_diff = b - a;
            }
        }
    }
    return max_diff;
}

double my_max_diff_d(obj_t* A, obj_t* B, dim_t m, dim_t n) {

    void*  buf_A    = bli_obj_buffer_at_off( A ); 
    inc_t  rs_A     = bli_obj_row_stride( A ); 
    inc_t  cs_A     = bli_obj_col_stride( A ); 
    double *buf_Aptr = buf_A;

    void*  buf_B    = bli_obj_buffer_at_off( B ); 
    inc_t  rs_B     = bli_obj_row_stride( B ); 
    inc_t  cs_B    = bli_obj_col_stride( B ); 
    double *buf_Bptr = buf_B;

    double max_diff = 0;
    for (dim_t i = 0; i < m; i++) {
        for (dim_t j = 0; j < n; j++) {
            double a = buf_Aptr[i*rs_A + j*cs_A];
            double b = buf_Bptr[i*rs_B + j*cs_B];
            if (a - b > max_diff) {
                max_diff = a - b;
            }
            else if (b - a > max_diff) {
                max_diff = b - a;
            }
        }
    }
    return max_diff;
}

int test_bli_strassen_ex( int m, int n, int k, int debug, fmm_t* fmm )
{   
    // fmm_t fmm = new_fmm("222.txt");
    // fmm_t fmm = new_fmm_ex("222.txt", 2);

    obj_t* null = 0;

    int failed = 0;

    num_t dt;
	//dim_t m, n, k;
	inc_t rsC, csC;
    inc_t rsA, csA;
    inc_t rsB, csB;
	side_t side;

	obj_t A, B, C, C_ref, diffM;
	obj_t* alpha;
	obj_t* beta;

    int    i, j;

    double tmp, error, flops;
    double ref_beg, ref_time, bl_dgemm_beg, bl_dgemm_time;
    int    nrepeats;
    int    lda, ldb, ldc, ldc_ref;
    double ref_rectime, bl_dgemm_rectime;
    double diff;

    dt = BLIS_DOUBLE;

    // rsC = 1; csC = m;
    // rsA = 1; csA = m;
    // rsB = 1; csB = k;

    rsC = n; csC = 1;
    rsA = k; csA = 1;
    rsB = n; csB = 1;

    bli_obj_create( dt, m, n, 0, 0, &C );
    bli_obj_create( dt, m, n, 0, 0, &C_ref );
    bli_obj_create( dt, m, n, 0, 0, &diffM );

	bli_obj_create( dt, m, k, 0, 0, &A );
	bli_obj_create( dt, k, n, 0, 0, &B );

	// Set the scalars to use.
	alpha = &BLIS_ONE;
	beta  = &BLIS_ONE;


#if 0
    bli_randm( &B );
    bli_randm( &A );
    bli_randm( &C );
    // bli_setm( &BLIS_ZERO, &C );
    bli_copym( &C, &C_ref );

    bli_printm( "matrix 'a', initialized by columns:", &A, "%5.3f", "" );
    bli_printm( "matrix 'b', initialized by columns:", &B, "%5.3f", "" );
    bli_printm( "matrix 'c', initialized by columns:", &C, "%5.3f", "" );
#else 
    // set matrices to known values for debug purposes. 

    bli_setm( &BLIS_ZERO, &B );
    bli_setd( &BLIS_MINUS_ONE, &B );
	bli_setm( &BLIS_ZERO, &C );
    bli_copym( &C, &C_ref );

    void*  buf_A    = bli_obj_buffer_at_off( &A ); 
	inc_t  rs_A     = bli_obj_row_stride( &A ); 
	inc_t  cs_A     = bli_obj_col_stride( &A ); 

    double *buf_Aptr = buf_A;

    for ( int p = 0; p < k; p ++ ) {
        for ( int i = 0; i < m; i ++ ) {
            buf_Aptr[i * rs_A + p * cs_A] = i;
        }
    }

    if (DEBUG_test) {
        bli_printm( "matrix 'a', initialized by columns:", &A, "%5.3f", "" );
        bli_printm( "matrix 'b', initialized by columns:", &B, "%5.3f", "" );
        bli_printm( "matrix 'c', initialized by columns:", &C, "%5.3f", "" );
    }
    else {
        // bli_printm( "matrix 'b', initialized by columns:", &B, "%5.3f", "" );
    }

#endif

    nrepeats = 1;

#if 0
    srand48 (time(NULL));

    // Randonly generate points in [ 0, 1 ].
    for ( p = 0; p < k; p ++ ) {
        for ( i = 0; i < m; i ++ ) {
            A( i, p ) = (double)( drand48() );
        }
    }
    for ( j = 0; j < n; j ++ ) {
        for ( p = 0; p < k; p ++ ) {
            B( p, j ) = (double)( drand48() );
        }
    }

    for ( j = 0; j < n; j ++ ) {
        for ( i = 0; i < m; i ++ ) {
            // C_ref( i, j ) = (double)( 0.0 );
            //     C( i, j ) = (double)( 0.0 );

            C_ref( i, j ) = (double)( drand48() );
            C( i, j )     = C_ref( i, j );

        }
    }
#endif

    for ( i = 0; i < nrepeats; i ++ ) {
        bl_dgemm_beg = bl_clock();
        {
            bli_fmm( alpha, &A, &B, beta, &C, fmm);
            // bli_strassen_ab( alpha, &A, &B, beta, &C);
        }
        bl_dgemm_time = bl_clock() - bl_dgemm_beg;

        if ( i == 0 ) {
            bl_dgemm_rectime = bl_dgemm_time;
        } else {
            bl_dgemm_rectime = bl_dgemm_time < bl_dgemm_rectime ? bl_dgemm_time : bl_dgemm_rectime;
        }
    }

    for ( i = 0; i < nrepeats; i ++ ) {
        ref_beg = bl_clock();
        {
            // bli_gemm( alpha, &A, &B, beta, &C_ref);
            my_mm(&A, &B, &C_ref, m, n, k);
        }
        ref_time = bl_clock() - ref_beg;

        if ( i == 0 ) {
            ref_rectime = ref_time;
        } else {
            ref_rectime = ref_time < ref_rectime ? ref_time : ref_rectime;
        }
    }

    double        resid, resid_other;
    obj_t  norm;
	double junk;

    bli_obj_scalar_init_detached( dt, &norm );

    if (DEBUG_test) {
        printf("\n\n----------------------\n\n");
        bli_printm( "RESULT 'C', initialized by columns:", &C, "%5.3f", "" );
        bli_printm( "REFERENCE 'C_REF', initialized by columns:", &C_ref, "%5.3f", "" );
    }

    bli_copym( &C_ref, &diffM );

    bli_subm( &C, &diffM );
	bli_normfm( &diffM, &norm );
	bli_getsc( &norm, &resid, &junk );

    // Compute overall floating point operations.
    flops = ( m * n / ( 1000.0 * 1000.0 * 1000.0 ) ) * ( 2 * k );

    if (debug)
        printf( "%5d\t %5d\t %5d\t %5.2lf\t %5.2lf\t %5.2g\t %5.2g\n",
                m, n, k, flops / bl_dgemm_rectime, flops / ref_rectime, resid, resid_other );
    else {
        if (resid > .0000001) {
            failed = 1;
            printf("\n\n");
            printf( "--> %5d\t %5d\t %5d\t %5.2lf\t %5.2lf\t %5.2g\t %5.2g\n",
                    m, n, k, flops / bl_dgemm_rectime, flops / ref_rectime, resid, resid_other );
            printf("\n\n");
        }
    }
    //printf( "%5d\t %5d\t %5d\t %5.2lf\n",
    //        m, n, k, flops / bl_dgemm_rectime );



    fflush(stdout);

	// Free the objects.
	bli_obj_free( &A );
	bli_obj_free( &B );
	bli_obj_free( &C );
    bli_obj_free( &C_ref );
    bli_obj_free( &diffM );

    return failed;
}

int test_bli_strassen(int m, int n, int k, int debug) {
    fmm_t fmm = new_fmm("classical.txt");
    int res = test_bli_strassen_ex(m, n, k, debug, &fmm);
    free_fmm(&fmm);
    return res;
}


int main( int argc, char *argv[] )
{

    fmm_t fmm = new_fmm_ex("111.txt", 1, -1, false, false);

    int m, n, k;
    // m = 2;
    // n = 11;
    // k = 2;
    m = 8;
    n = 8;
    k = 8;

    printf("\n\n\n");

    test_bli_strassen_ex( m, n, k, 1, &fmm);

    free_fmm(&fmm);

    return 0;
}
