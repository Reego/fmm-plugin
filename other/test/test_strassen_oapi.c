#include "bl_dgemm.h"
#include <time.h>



void test_bli_strassen( int m, int n, int k )
{
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

    bli_obj_create( dt, m, n, rsC, csC, &C );
    bli_obj_create( dt, m, n, rsC, csC, &C_ref );
    bli_obj_create( dt, m, n, rsC, csC, &diffM );

	bli_obj_create( dt, m, k, rsA, csA, &A );
	bli_obj_create( dt, k, n, rsB, csB, &B );

	// Set the scalars to use.
	alpha = &BLIS_ONE;
	beta  = &BLIS_ONE;

#if 0

    A    = (double*)malloc( sizeof(double) * m * k );
    B    = (double*)malloc( sizeof(double) * k * n );

    lda     = m;
    ldb     = k;
    ldc     = ( ( m - 1 ) / DGEMM_MR + 1 ) * DGEMM_MR;
    ldc_ref = ( ( m - 1 ) / DGEMM_MR + 1 ) * DGEMM_MR;
    C     = bl_malloc_aligned( ldc, n + 4, sizeof(double) );
    C_ref = bl_malloc_aligned( ldc, n + 4, sizeof(double) );

#endif


#if 1
	bli_randm( &A );
    bli_randm( &B );
    bli_randm( &C );
    bli_copym( &C, &C_ref );
#else 
    // set matrices to known values for debug purposes. 

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

#endif 

#if 0
	bli_printm( "a: randomized", &A, "%7.4f", "" );
	// bli_printm( "b: set to 1.0", &B, "%4.1f", "" );
	bli_printm( "c: initial value", &C, "%4.1f", "" );
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

#if 0 
    //wrote this part to test the linear combination + packing routine. 
    dim_t ks = 2; 
    dim_t k_part = k;

    obj_t  Alarge, Blarge;

    bli_obj_create( dt, m, k*ks, rsA*ks, csA, &Alarge );
    bli_randm( &Alarge );

    bli_obj_create( dt, k*ks, n, rsB, csB, &Blarge );
    bli_randm( &Blarge );


    obj_t A0, A1;
    bli_acquire_mpart_ndim( BLIS_FWD, BLIS_SUBPART0,
		                        k_part, k_part, &Alarge, &A0 );
	bli_acquire_mpart_ndim( BLIS_FWD, BLIS_SUBPART1,
		                        k_part, k_part, &Alarge, &A1 );


    obj_t B0, B1;
    bli_acquire_mpart_mdim( BLIS_FWD, BLIS_SUBPART0, 
                                k_part, k_part, &Blarge, &B0 );
	bli_acquire_mpart_mdim( BLIS_FWD, BLIS_SUBPART1, 
                                k_part, k_part, &Blarge, &B1 );

    dim_t m0, m1, k0, k1; 
    dim_t offm0, offk0, offm1, offk1;

    m0 = bli_obj_length( &A0 ); 
    k0 = bli_obj_width( &A0 ); 

    m1 = bli_obj_length( &A1 ); 
    k1 = bli_obj_width( &A1 ); 

    offm0 = bli_obj_off( BLIS_M, &A0 );
    offk0 = bli_obj_off( BLIS_N, &A0 );

    offm1 = bli_obj_off( BLIS_M, &A1 );
    offk1 = bli_obj_off( BLIS_N, &A1 );

    obj_t Aadd;
    bli_obj_create( dt, m, k_part, rsA, csA, &Aadd );
    bli_copym( &A0 , &Aadd);

    obj_t Badd;
    bli_obj_create( dt, k_part, n, rsB, csB, &Badd );
    bli_copym( &B0 , &Badd);
   
#endif


    for ( i = 0; i < nrepeats; i ++ ) {
        bl_dgemm_beg = bl_clock();
        {
            bli_strassen_ab( alpha, &A, &B, beta, &C );
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
            bli_gemm( alpha, &A, &B, beta, &C_ref);
        }
        ref_time = bl_clock() - ref_beg;

        if ( i == 0 ) {
            ref_rectime = ref_time;
        } else {
            ref_rectime = ref_time < ref_rectime ? ref_time : ref_rectime;
        }
    }

#if 0

    bli_printm( "C: after value", &C, "%4.1f", "" );

    bli_printm( "C_ref: after value", &C_ref, "%4.1f", "" );
#endif 

    double        resid;
    obj_t  norm;
	double junk;

    bli_obj_scalar_init_detached( dt, &norm );


    bli_copym( &C_ref, &diffM );

    bli_subm( &C, &diffM );
	bli_normfm( &diffM, &norm );
	bli_getsc( &norm, &resid, &junk );

    // Compute overall floating point operations.
    flops = ( m * n / ( 1000.0 * 1000.0 * 1000.0 ) ) * ( 2 * k );

    printf( "%5d\t %5d\t %5d\t %5.2lf\t %5.2lf\t %5.2g\n",
            m, n, k, flops / bl_dgemm_rectime, flops / ref_rectime, resid );
    //printf( "%5d\t %5d\t %5d\t %5.2lf\n",
    //        m, n, k, flops / bl_dgemm_rectime );



    fflush(stdout);

	// Free the objects.
	bli_obj_free( &A );
	bli_obj_free( &B );
	bli_obj_free( &C );
    bli_obj_free( &C_ref );

}
