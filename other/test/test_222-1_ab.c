#include "bl_dgemm.h"
#include <time.h>



void test_bl_dgemm( int m, int n, int k )
{
    int    i, j, p, nx;
    double *A, *B, *C, *C_ref;
    double tmp, error, flops;
    double ref_beg, ref_time, bl_dgemm_beg, bl_dgemm_time;
    int    nrepeats;
    int    lda, ldb, ldc, ldc_ref;
    double ref_rectime, bl_dgemm_rectime;
    double diff;

    A    = (double*)malloc( sizeof(double) * m * k );
    B    = (double*)malloc( sizeof(double) * k * n );

    lda     = m;
    ldb     = k;
    ldc     = ( ( m - 1 ) / DGEMM_MR + 1 ) * DGEMM_MR;
    ldc_ref = ( ( m - 1 ) / DGEMM_MR + 1 ) * DGEMM_MR;
    C     = bl_malloc_aligned( ldc, n + 4, sizeof(double) );
    C_ref = bl_malloc_aligned( ldc, n + 4, sizeof(double) );

    nrepeats = 1;

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

    for ( i = 0; i < nrepeats; i ++ ) {
        bl_dgemm_beg = bl_clock();
        {
            bl_dgemm_strassen_ab( m, n, k, A, lda, B, ldb, C, ldc );
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
#if 1
            char transA = 'N', transB = 'N';
            double alpha = 1.0, beta = 1.0;
            dgemm_( &transA, &transB, &m, &n, &k, &alpha,
                    A, &lda, B, &ldb, &beta, C_ref, &ldc_ref );
#else
            bl_dgemm_ref( m, n, k, A, lda, B, ldb, C_ref, ldc_ref );
#endif
        }
        ref_time = bl_clock() - ref_beg;

        if ( i == 0 ) {
            ref_rectime = ref_time;
        } else {
            ref_rectime = ref_time < ref_rectime ? ref_time : ref_rectime;
        }
    }

    diff = bl_compare_error( ldc, ldc_ref, m, n, C, C_ref );

    // Compute overall floating point operations.
    flops = ( m * n / ( 1000.0 * 1000.0 * 1000.0 ) ) * ( 2 * k );

    printf( "%5d\t %5d\t %5d\t %5.2lf\t %5.2lf\t %5.2lf\n",
            m, n, k, flops / bl_dgemm_rectime, flops / ref_rectime, diff );
    //printf( "%5d\t %5d\t %5d\t %5.2lf\n",
    //        m, n, k, flops / bl_dgemm_rectime );



    fflush(stdout);

    free( A     );
    free( B     );
    free( C     );
    free( C_ref );
}

int main( int argc, char *argv[] )
{
    int    m, n, k;

    if ( argc != 4 ) {
        printf( "Error: require 3 arguments, but only %d provided.\n", argc - 1 );
        exit( 0 );
    }

    sscanf( argv[ 1 ], "%d", &m );
    sscanf( argv[ 2 ], "%d", &n );
    sscanf( argv[ 3 ], "%d", &k );

#if 0
    test_bl_dgemm( m, n, k );
#else
    test_bli_strassen( m, n, k );
#endif

    return 0;
}
