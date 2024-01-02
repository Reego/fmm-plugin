#include "blis.h"
#include "bli_fmm.h"
#include <time.h>
#include <complex.h>

#define DEBUG_strassen 0

void bl_acquire_spart 
     (
             dim_t     row_splits,
             dim_t     col_splits,
             dim_t     split_rowidx,
             dim_t     split_colidx,
       const obj_t*    obj, // source
             obj_t*    sub_obj // destination
     )

{
    dim_t m, n;
    dim_t row_part, col_part; //size of partition
    dim_t row_left, col_left; // edge case
    inc_t  offm_inc = 0;
	inc_t  offn_inc = 0;

    m = bli_obj_length( obj ); 
    n = bli_obj_width( obj ); 

    row_part = m / row_splits;
    col_part = n / col_splits;

    row_left = m % row_splits;
    col_left = n % col_splits;

    /* AT the moment not dealing with edge cases. bli_strassen_ab checks for 
    edge cases. But does not do anything with it. 
    */
    if ( 0 && row_left != 0 || col_left != 0 && 0) {
        bli_abort();
    }

    row_part = (m+1) / row_splits;
    col_part = (n+1) / col_splits;

    // printf("\n\nrow_part %d col_part %d\n\n\n", row_part, col_part);

    bli_obj_init_subpart_from( obj, sub_obj );

    bli_obj_set_dims( row_part, col_part, sub_obj );

    offm_inc = split_rowidx * row_part;
	offn_inc = split_colidx * col_part;

    //Taken directly from BLIS. Need to verify if this is still true. 
    // Compute the diagonal offset based on the m and n offsets.
	doff_t diagoff_inc = ( doff_t )offm_inc - ( doff_t )offn_inc;

    bli_obj_inc_offs( offm_inc, offn_inc, sub_obj );
	bli_obj_inc_diag_offset( diagoff_inc, sub_obj );

}

void bli_strassen_ab( obj_t* alpha, obj_t* A, obj_t* B, obj_t* beta, obj_t* C )
{
    static int registered = false;

    bli_init_once();

    if (!registered) {
        err_t err = bli_plugin_register_fmm_blis();
        if (err != BLIS_SUCCESS)
        {
            printf("error %d\n",err);
            bli_abort();
        }
        registered = true;
    }

    cntx_t* cntx = NULL;
    rntm_t* rntm = NULL;
    
	// Check the operands.
	// if ( bli_error_checking_is_enabled() )
	// 	bli_gemm_check( alpha, A, B, beta, C, cntx );

	// Check for zero dimensions, alpha == 0, or other conditions which
	// mean that we don't actually have to perform a full l3 operation.
	if ( bli_l3_return_early_if_trivial( alpha, A, B, beta, C ) == BLIS_SUCCESS )
		return;

	// Default to using native execution.
	num_t dt = bli_obj_dt( C );
	ind_t im = BLIS_NAT;

	// If necessary, obtain a valid context from the gks using the induced
	// method id determined above.
	if ( cntx == NULL ) cntx = bli_gks_query_cntx();

	// Alias A, B, and C in case we need to apply transformations.
	obj_t A_local;
	obj_t B_local;
	obj_t C_local;

    dim_t m, k, n;
    obj_t A0, B0, C0;

    m = bli_obj_length( C );
    n = bli_obj_width( C );
    k = bli_obj_width( A );

    dim_t m_edge, m_whole, k_edge, k_whole, n_edge, n_whole;
    dim_t m_splits, k_splits, n_splits;

    // For <2, 2, 2> Strassen -> This needs to be made generic. 
    m_splits = 2, k_splits = 2, n_splits = 2;

    m_edge = m % ( m_splits * DGEMM_MR );
    k_edge = k % ( k_splits );
    n_edge = n % ( n_splits * DGEMM_NR );
    m_whole = (m - m_edge);
    k_whole = (k - k_edge); 
    n_whole = (n - n_edge);

    //printf("m = %d, k = %d, n = %d\n", m, k, n);
    //printf("m_edge = %d, k_edge = %d, n_edge = %d\n", m_edge, k_edge, n_edge);

    //printf("m_whole = %d, k_whole = %d, n_whole %d\n", m_whole, k_whole, n_whole);

    bl_acquire_spart (m_splits, k_splits, 0, 0, A, &A0 );
    bl_acquire_spart (k_splits, n_splits, 0, 0, B, &B0 );
    bl_acquire_spart (m_splits, n_splits, 0, 0, C, &C0 );

#if 1
    bli_obj_alias_submatrix( &A0, &A_local );
	bli_obj_alias_submatrix( &B0, &B_local );
	bli_obj_alias_submatrix( &C0, &C_local );
#else
    bli_obj_alias_submatrix( A, &A_local );
	bli_obj_alias_submatrix( B, &B_local );
	bli_obj_alias_submatrix( C, &C_local );
#endif
	gemm_cntl_t cntl;
	bli_gemm_cntl_init
	(
	  im,
	  BLIS_GEMM,
	  alpha,
	  &A_local,
	  &B_local,
	  beta,
	  &C_local,
	  cntx,
	  &cntl
	);

    fmm_params_t paramsA, paramsB, paramsC;
    // paramsA.id = 1;
    // paramsB.id = 2;
    // paramsC.id = 3;
    paramsA.m_max = m; paramsA.n_max = k;
    paramsB.m_max = n; paramsB.n_max = k;
    paramsC.m_max = n; paramsC.n_max = m;
    // printf("\nPARAMS A %d %d\n\n", m, k);
    // printf("\nPARAMS B %d %d\n\n", k, n);

#if 1
    func_t *pack_ukr;

    pack_ukr = bli_cntx_get_ukrs( FMM_BLIS_PACK_UKR, cntx );
    bli_gemm_cntl_set_packa_ukr_simple( pack_ukr , &cntl );
    bli_gemm_cntl_set_packb_ukr_simple( bli_cntx_get_ukrs( FMM_BLIS_PACK_UKR, cntx ), &cntl );
    bli_gemm_cntl_set_ukr_simple( bli_cntx_get_ukrs( FMM_BLIS_GEMM_UKR, cntx ), &cntl );

    bli_gemm_cntl_set_packa_params((const void *) &paramsB, &cntl);
    bli_gemm_cntl_set_packb_params((const void *) &paramsA, &cntl);
    bli_gemm_cntl_set_params((const void *) &paramsC, &cntl);
#endif

    // There is probably a better way to define these...?? I don't like this. 
    /*
    1 0 0 0 1 0 0
    1 0 -1 -1 0 -1 0
    0 -1 0 0 1 1 -1
    0 0 -1 0 0 0 -1
    #
    1 0 0 -1 1 -1 0
    0 -1 0 0 1 0 0
    0 0 -1 1 0 0 0
    0 1 -1 0 0 -1 1
    #
    1 0 0 -1 0 0 0
    -1 -1 0 0 1 1 0
    0 0 1 1 0 -1 1
    0 1 0 0 0 0 -1
    */
    int U[4][7] = {{ 1,  0,  0,  0,  1,  0,  0}, 
                   { 1,  0, -1, -1,  0, -1,  0}, 
                   { 0, -1,  0,  0,  1,  1, -1}, 
                   { 0,  0, -1,  0,  0,  0, -1} };

    int V[4][7] = { { 1,  0,  0, -1,  1, -1,  0}, 
                    { 0, -1,  0,  0,  1,  0,  0}, 
                    { 0,  0, -1,  1,  0,  0,  0}, 
                    { 0,  1, -1,  0,  0, -1,  1} } ;

    int W[4][7] = {{ 1,  0,  0, -1,  0,  0,  0}, 
                   {-1, -1,  0,  0,  1,  1,  0},
                   { 0,  0,  1,  1,  0, -1,  1}, 
                   { 0,  1,  0,  0,  0, 0,  -1}};
    
    // For Matrix A
    // (0,0)
    // (0, k/2)
    // (m/2, 0)
    // (m/2, k/2)

    m_whole = m;
    n_whole = n;
    k_whole = k;

    dim_t row_off_A[4], col_off_A[4];

    row_off_A[0] = 0,         col_off_A[0] = 0;
    row_off_A[1] = 0,         col_off_A[1] = (k_whole+1)/2;
    row_off_A[2] = (m_whole+1)/2, col_off_A[2] = 0;
    row_off_A[3] = (m_whole+1)/2, col_off_A[3] = (k_whole+1)/2;


    // For Matrix B
    // (0, 0)
    // (0, n/2)
    // (k/2, 0)
    // (k/2, n/2)

    dim_t row_off_B[4], col_off_B[4];

    // row_off_B[0] = 0,         col_off_B[0] = 0;
    // row_off_B[1] = 0,         col_off_B[1] = (n_whole+1)/2;
    // row_off_B[2] = (k_whole+1)/2, col_off_B[2] = 0;
    // row_off_B[3] = (k_whole+1)/2, col_off_B[3] = (n_whole+1)/2;

    row_off_B[0] = 0,                       col_off_B[0] = 0;
    row_off_B[1] = (n_whole+1)/2,           col_off_B[1] = 0;
    row_off_B[2] = 0,                       col_off_B[2] = (k_whole+1)/2;
    row_off_B[3] = (n_whole+1)/2,           col_off_B[3] = (k_whole+1)/2;

    // For Matrix C
    // (0, 0)
    // (0, n/2)
    // (m/2, 0)
    // (m/2, n/2)

    dim_t row_off_C[4], col_off_C[4];

    row_off_C[0] = 0,         col_off_C[0] = 0;
    row_off_C[1] = (n_whole+1)/2,         col_off_C[1] = 0;
    row_off_C[2] = 0, col_off_C[2] = (m_whole+1)/2;
    row_off_C[3] = (n_whole+1)/2, col_off_C[3] = (m_whole+1)/2;

    // printf("%d %d | %d \n", m_whole, k_whole, m_edge;

    // if (false)printf("\n\nSET UP %d %d %d\n\n\n", m, n, k);

    for ( dim_t r = 0; r < FMM_BLIS_MULTS; r++ )
    {

        paramsA.nsplit = 0;
        paramsB.nsplit = 0;
        paramsC.nsplit = 0;

        //Zeroing out the elements before calling the next gemm.
        for (dim_t i = 0; i < 4; i++)
        {
            paramsA.coef[i] = 0;
            paramsB.coef[i] = 0;
            paramsC.coef[i] = 0;

            paramsA.off_m[i] = 0;
            paramsA.off_n[i] = 0;

            paramsB.off_m[i] = 0;
            paramsB.off_n[i] = 0;

            paramsC.off_m[i] = 0;
            paramsC.off_n[i] = 0;
        }

        for (dim_t isplits = 0; isplits < 4; isplits++)
        {
            if(U[isplits][r] != 0)
            {
                ((float*)paramsA.coef)[paramsA.nsplit] = U[isplits][r];
                // printf("\n\nHEY %f -- %d\n\n", creal(
                    // ((double complex*)paramsA.coef)[isplits]
                    // ), paramsA.nsplit);
            }
            else {
                // printf("\n\nOHH %f\n\n", creal((((complex double)*)paramsA.coef)[isplits]));
            }

            if(V[isplits][r] != 0)
            {
                //(when packing, m is the "short micro-panel dimension (m or n)
	            // ", and n is the "long micro-panel dimension (k)")
                // paramsB.coef[paramsB.nsplit] = V[isplits][r];
                ((float*)paramsB.coef)[paramsB.nsplit] = V[isplits][r];
            }
            if(W[isplits][r] != 0)
            {
                // paramsC.coef[paramsC.nsplit] = W[isplits][r];
                ((float*)paramsC.coef)[paramsC.nsplit] = W[isplits][r];
            }

            paramsA.off_m[paramsA.nsplit] = row_off_A[isplits];
            paramsA.off_n[paramsA.nsplit] = col_off_A[isplits];
            paramsA.nsplit++;

            paramsB.off_m[paramsB.nsplit] = row_off_B[isplits];
            paramsB.off_n[paramsB.nsplit] = col_off_B[isplits];
            paramsB.nsplit++;

            paramsC.off_m[paramsC.nsplit] = row_off_C[isplits];
            paramsC.off_n[paramsC.nsplit] = col_off_C[isplits];
            paramsC.nsplit++;
        }

        // printf("\n\nHEY %d %p\n\n", ((double*)paramsA.coef)[0], paramsB.coef);
        // printf("\n\nparamsB.coef %p\n\n", ((double*)paramsB.coef));
        // printf("paramsA %5.3f paramsB %5.3f", ((double*)paramsA.coef)[0], ((double*)paramsB.coef)[0]);

        // printf("paramsA %5.3f paramsB %5.3f", ((double*)paramsA.coef)[0], ((double*)paramsB.coef)[0]);
	    // Invoke the internal back-end via the thread handler.
        if (DEBUG_strassen)
            printf("\n\n==============MULT=============\n\n");

	    bli_l3_thread_decorator
	    (
            &A_local,
            &B_local,
            &C_local,
            cntx,
            ( cntl_t* )&cntl,
            rntm
        );
        // if (1) break;
    }

}
