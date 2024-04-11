// zeroing out packing buffer

#include "blis.h"
#include "bli_fmm.h"
#include <time.h>

#define X 15
#define N_CONST X
#define M_CONST X
#define K_CONST X

#define DEBUG_test 0
#define RUN_trials 1
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

int test_bli_strassen_ex( int m, int n, int k, int debug )
{   
    fmm_t fmm = new_fmm("classical.txt");

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


#if 1
    bli_randm( &B );
    bli_randm( &A );
    bli_randm( &C );
    bli_copym( &C, &C_ref );

    // bli_printm( "matrix 'a', initialized by columns:", &A, "%5.3f", "" );
    // bli_printm( "matrix 'b', initialized by columns:", &B, "%5.3f", "" );
    // bli_printm( "matrix 'c', initialized by columns:", &C, "%5.3f", "" );
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
            bli_strassen_ab_ex( alpha, &A, &B, beta, &C, fmm);
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

    if (!debug)
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

    free_fmm(fmm);

    return failed;
}

int test_bli_symm_strassen_ex( int m, int n, int k, int debug )
{   
    k = m;
    fmm_t fmm = new_fmm("classical.txt");

    obj_t* null = 0;

    int failed = 0;

    num_t dt;
    //dim_t m, n, k;
    inc_t rsC, csC;
    inc_t rsA, csA;
    inc_t rsB, csB;

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

    dt = BLIS_DCOMPLEX;

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

    side_t side = BLIS_LEFT;


#if 1
    bli_randm( &B );
    bli_randm( &A );
    bli_randm( &C );
    bli_copym( &C, &C_ref );

    bli_setm( &BLIS_ZERO, &A ); // ###

    bli_obj_set_struc( BLIS_SYMMETRIC, &A ); // ###
    bli_obj_set_uplo( BLIS_UPPER, &A ); // ###
    bli_randm( &A ); // ###

    // c := beta * c + alpha * a * b, where 'a' is symmetric and upper-stored.
    // Note that the first 'side' operand indicates the side from which matrix
    // 'a' is multiplied into 'b'

    // bli_strassen_ab(alpha, &a, &b, beta, &c_ref );

    // bli_printm( "matrix 'a', initialized by columns:", &A, "%5.3f", "" );
    // bli_printm( "matrix 'b', initialized by columns:", &B, "%5.3f", "" );
    // bli_printm( "matrix 'c', initialized by columns:", &C, "%5.3f", "" );
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
            bli_strassen_ab_symm( alpha, &A, &B, beta, &C );
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
            bli_symm(side, alpha, &A, &B, beta, &C_ref);
            // my_mm(&A, &B, &C_ref, m, n, k);
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

    if (dt == BLIS_DCOMPLEX) {
        resid = my_max_diff_d(&C, &C_ref, m, n);
    }
    else {
        bli_copym( &C_ref, &diffM );
        bli_subm( &C, &diffM );
        bli_normfm( &diffM, &norm );
        bli_getsc( &norm, &resid, &junk );
    }

    // Compute overall floating point operations.
    flops = ( m * n / ( 1000.0 * 1000.0 * 1000.0 ) ) * ( 2 * k );

    if (!debug)
        printf( "%5d\t %5d\t %5d\t %5.2lf\t %5.2lf\t resid %5.2g\t other %5.2g\n",
                m, n, k, flops / bl_dgemm_rectime, flops / ref_rectime, resid, resid_other );
        if (resid > .0000001) {
            failed = 1;
        }
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

    free_fmm(fmm);

    return failed;
}

void test_bli_strassen( int m, int n, int k )
{   test_bli_strassen_ex(m, n, k, true);
}

int other2()
{
    num_t dt;
    dim_t m, n, k;
    inc_t rs, cs;
    side_t side;

    obj_t a, b, c, c_ref;
    obj_t* alpha;
    obj_t* beta;


    //
    // This file demonstrates level-3 operations.
    //

    //
    // Example 3: Perform a symmetric matrix-matrix multiply (symm) operation.
    //

    printf( "\n#\n#  -- Example 3 --\n#\n\n" );

    // Create some matrix and vector operands to work with.
    dt = BLIS_DCOMPLEX;
    m = 5; n = 6; rs = 0; cs = 0;
    // m = 5; n = 5; rs = 0; cs = 0;
    bli_obj_create( dt, m, m, rs, cs, &a );
    bli_obj_create( dt, m, n, rs, cs, &b );
    bli_obj_create( dt, m, n, rs, cs, &c );
    bli_obj_create( dt, m, n, rs, cs, &c_ref );

    printf("C dt is %d", bli_obj_dt( &c ));

    // Set the scalars to use.
    alpha = &BLIS_ONE;
    beta  = &BLIS_ONE;

    // Set the side operand.
    side = BLIS_LEFT;

    // Initialize matrices 'b' and 'c'.
    // bli_setm( &BLIS_ONE,  &a ); // ###


    bli_setm( &BLIS_ONE,  &b );
    bli_setm( &BLIS_ZERO, &c );
    bli_setm( &BLIS_ZERO, &c_ref );

    // Zero out all of matrix 'a'. This is optional, but will avoid possibly
    // displaying junk values in the unstored triangle.
    bli_setm( &BLIS_ZERO, &a ); // ###

    // Mark matrix 'a' as symmetric and stored in the upper triangle, and
    // then randomize that upper triangle.


    bli_obj_set_struc( BLIS_SYMMETRIC, &a ); // ###
    bli_obj_set_uplo( BLIS_UPPER, &a ); // ###
    bli_randm( &a ); // ###

    bli_printm( "a: randomized (zeros in lower triangle)", &a, "%4.1f", "" );
    bli_printm( "b: set to 1.0", &b, "%4.1f", "" );
    bli_printm( "c: initial value", &c, "%4.1f", "" );

    // c := beta * c + alpha * a * b, where 'a' is symmetric and upper-stored.
    // Note that the first 'side' operand indicates the side from which matrix
    // 'a' is multiplied into 'b'.
    bli_strassen_ab_symm( alpha, &a, &b, beta, &c );
    bli_printm( "c: after symm", &c, "%4.1f", "" );

    // bli_strassen_ab(alpha, &a, &b, beta, &c_ref );
    bli_symm(side, alpha, &a, &b, beta, &c_ref);
    bli_printm( "c_ref: after symm", &c_ref, "%4.1f", "" );

    // Free the objects.
    bli_obj_free( &a );
    bli_obj_free( &b );
    bli_obj_free( &c );
    bli_obj_free( &c_ref );


    return 0;
}


int other()
{
    num_t dt;
    dim_t m, n, k;
    inc_t rs, cs;
    side_t side;

    obj_t a, b, c, c_ref;
    obj_t* alpha;
    obj_t* beta;


    //
    // This file demonstrates level-3 operations.
    //

    //
    // Example 3: Perform a symmetric matrix-matrix multiply (symm) operation.
    //

    printf( "\n#\n#  -- Example 3 --\n#\n\n" );

    // Create some matrix and vector operands to work with.
    dt = BLIS_DCOMPLEX;
    // dt = BLIS_DOUBLE;
    m = 4; n = 4; rs = 0; cs = 0;
    k = m;
    // m = 5; n = 5; rs = 0; cs = 0;
    bli_obj_create( dt, m, k, rs, cs, &a );
    bli_obj_create( dt, k, n, rs, cs, &b );
    bli_obj_create( dt, m, n, rs, cs, &c );
    bli_obj_create( dt, m, n, rs, cs, &c_ref );

    // Set the scalars to use.
    alpha = &BLIS_ONE;
    beta  = &BLIS_ONE;

    // Set the side operand.
    side = BLIS_LEFT;

    // Initialize matrices 'b' and 'c'.
    // bli_setm( &BLIS_ONE,  &a ); // ###


    bli_setm( &BLIS_ONE,  &b );
    bli_setm( &BLIS_ZERO, &c );
    bli_setm( &BLIS_ZERO, &c_ref );

    // Zero out all of matrix 'a'. This is optional, but will avoid possibly
    // displaying junk values in the unstored triangle.
    bli_setm( &BLIS_ZERO, &a ); // ###

    // Mark matrix 'a' as symmetric and stored in the upper triangle, and
    // then randomize that upper triangle.


    bli_obj_set_struc( BLIS_SYMMETRIC, &a ); // ###
    bli_obj_set_uplo( BLIS_UPPER, &a ); // ###
    bli_randm( &a ); // ###

    bli_printm( "a: randomized (zeros in lower triangle)", &a, "%4.1f", "" );
    bli_printm( "b: set to 1.0", &b, "%4.1f", "" );
    bli_printm( "c: initial value", &c, "%4.1f", "" );

    // c := beta * c + alpha * a * b, where 'a' is symmetric and upper-stored.
    // Note that the first 'side' operand indicates the side from which matrix
    // 'a' is multiplied into 'b'.
    bli_strassen_ab_symm( alpha, &a, &b, beta, &c );
    bli_printm( "c: after symm", &c, "%4.1f", "" );

    // bli_gemm(alpha, &a, &b, beta, &c_ref );
    bli_symm(side, alpha, &a, &b, beta, &c_ref);
    bli_printm( "c_ref: after symm", &c_ref, "%4.1f", "" );

    // Free the objects.
    bli_obj_free( &a );
    bli_obj_free( &b );
    bli_obj_free( &c );
    bli_obj_free( &c_ref );


    return 0;
}

// -----------------------------------------------------------------------------



int main( int argc, char *argv[] )
{

    if (0) {
        other();
        return 0;
    }

    #if RUN_trials
    int LIMIT = 1204;
    int START = 400;
    int END = 404;

    int failed = 0;

    // for (int l = 0; l < LIMIT; l++) {
        for (int m = START; m < END; m++) {
            for (int n = START; n < END; n++) {
                for (int k = START; k < END; k++) {
                    if (m + n + k <= LIMIT) {
                        #if SQUARE
                        if (m == n && n == k)
                            failed += test_bli_symm_strassen_ex(m, n, k, 0);
                        #else
                            failed += test_bli_symm_strassen_ex(m, n, k, 0);
                        #endif
                    }
                }
            }
        }

    printf("\n\n\n=======================\nFailed %d tests\n\n", failed);
    // }
    #else

    int m, n, k;
    m = 2; n = 2; k = 6;

    test_bli_symm_strassen_ex( m, n, k, 0);

    // test_bli_strassen_ex( m, n, k, 1);
    #endif

    // #if 0
    // test_bl_dgemm( m, n, k );
    // #else
    // test_bli_strassen( m, n, k );
    // #endif



    // if ( argc != 4 ) {
    //     printf( "Error: require 3 arguments, but only %d provided.\n", argc - 1 );
    //     exit( 0 );
    // }

    // sscanf( argv[ 1 ], "%d", &m );
    // sscanf( argv[ 2 ], "%d", &n );
    // sscanf( argv[ 3 ], "%d", &k );

    return 0;
}
