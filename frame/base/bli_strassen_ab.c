#include "blis.h"
#include "bli_fmm.h"
#include <time.h>
#include <complex.h>

#define _U( i,j ) fmm.U[ (i)*fmm.R + (j) ]
#define _V( i,j ) fmm.V[ (i)*fmm.R + (j) ]
#define _W( i,j ) fmm.W[ (i)*fmm.R + (j) ]


/* Define Strassen's algorithm */
int STRASSEN_FMM_U[4][7] = {{1, 0, 1, 0, 1, -1, 0}, {0, 0, 0, 0, 1, 0, 1}, {0, 1, 0, 0, 0, 1, 0}, {1, 1, 0, 1, 0, 0, -1}};
int STRASSEN_FMM_V[4][7] = {{1, 1, 0, -1, 0, 1, 0}, {0, 0, 1, 0, 0, 1, 0}, {0, 0, 0, 1, 0, 0, 1}, {1, 0, -1, 0, 1, 0, 1}};
int STRASSEN_FMM_W[4][7] = {{1, 0, 0, 1, -1, 0, 1}, {0, 0, 1, 0, 1, 0, 0}, {0, 1, 0, 1, 0, 0, 0}, {1, -1, 1, 0, 0, 1, 0}};

fmm_t STRASSEN_FMM = {
    .m_tilde = 2,
    .n_tilde = 2,
    .k_tilde = 2,
    .R = 7,
    .U = &STRASSEN_FMM_U,
    .V = &STRASSEN_FMM_V,
    .W = &STRASSEN_FMM_W,
};

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

    row_part = m / row_splits;
    col_part = n / col_splits;

    if (m % row_splits != 0) {
        ++row_part;
    }

    if (n % col_splits != 0) {
        ++col_part;
    }

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

void init_part_offsets(dim_t* row_off, dim_t* col_off, dim_t* part_m, dim_t* part_n, dim_t row_whole, dim_t col_whole, int row_tilde, int col_tilde) {

    int num_row_part_whole = row_whole % row_tilde;
    if (row_whole % row_tilde == 0) num_row_part_whole = 0;
    dim_t row_part_size = row_whole / row_tilde;

    int num_col_part_whole = col_whole % col_tilde;
    if (col_whole % col_tilde == 0) num_col_part_whole = 0;
    dim_t col_part_size = col_whole / col_tilde;

    for (int i = 0; i < row_tilde; i++) {
        for (int j = 0; j < col_tilde; j++) {
            int part_index = j + i * col_tilde;

            int whole_i = bli_min(i, num_row_part_whole);
            int partial_i = bli_min(row_tilde - num_row_part_whole, i - whole_i);

            int whole_j = bli_min(j, num_col_part_whole);
            int partial_j = bli_min(col_tilde - num_col_part_whole, j - whole_j);

            if (i < num_row_part_whole)
                part_m[part_index] = row_part_size + 1;
            else
                part_m[part_index] = row_part_size;

            if (j < num_col_part_whole)
                part_n[part_index] = col_part_size + 1;
            else
                part_n[part_index] = col_part_size;

            row_off[part_index] = whole_i * (row_part_size + 1) + partial_i * row_part_size;

            col_off[part_index] = whole_j * (col_part_size + 1) + partial_j * col_part_size;
        }
    }
}

void bli_strassen_ab_ex( obj_t* alpha, obj_t* A, obj_t* B, obj_t* beta, obj_t* C, fmm_t fmm) {

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
    if ( bli_error_checking_is_enabled() )
     bli_gemm_check( alpha, A, B, beta, C, cntx );

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

    const int M_TILDE = fmm.m_tilde;
    const int N_TILDE = fmm.n_tilde;
    const int K_TILDE = fmm.k_tilde;

    m_splits = M_TILDE, k_splits = K_TILDE, n_splits = N_TILDE;

    m_edge = m % ( m_splits * DGEMM_MR );
    k_edge = k % ( k_splits );
    n_edge = n % ( n_splits * DGEMM_NR );
    m_whole = (m - m_edge);
    k_whole = (k - k_edge); 
    n_whole = (n - n_edge);

    bl_acquire_spart (m_splits, k_splits, 0, 0, A, &A0 );
    bl_acquire_spart (k_splits, n_splits, 0, 0, B, &B0 );
    bl_acquire_spart (m_splits, n_splits, 0, 0, C, &C0 );

    bli_obj_alias_submatrix( &A0, &A_local );
    bli_obj_alias_submatrix( &B0, &B_local );
    bli_obj_alias_submatrix( &C0, &C_local );

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

    paramsA.m_max = m; paramsA.n_max = k;
    paramsB.m_max = n; paramsB.n_max = k;
    paramsC.m_max = n; paramsC.n_max = m;
    paramsC.local = &C_local;

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

    m_whole = m;
    n_whole = n;
    k_whole = k;

    dim_t row_off_A[M_TILDE * K_TILDE], col_off_A[M_TILDE * K_TILDE];
    dim_t part_m_A[M_TILDE * K_TILDE], part_n_A[M_TILDE * K_TILDE];

    init_part_offsets(row_off_A, col_off_A, part_m_A, part_n_A, m_whole, k_whole, M_TILDE, K_TILDE);

    dim_t row_off_B[K_TILDE * N_TILDE], col_off_B[K_TILDE * N_TILDE];
    dim_t part_m_B[K_TILDE * N_TILDE], part_n_B[K_TILDE * N_TILDE];

    init_part_offsets(col_off_B, row_off_B, part_n_B, part_m_B, k_whole, n_whole, K_TILDE, N_TILDE); // since B is transposed... something idk.

    dim_t row_off_C[M_TILDE * N_TILDE], col_off_C[M_TILDE * N_TILDE];
    dim_t part_m_C[M_TILDE * N_TILDE], part_n_C[M_TILDE * N_TILDE];

    init_part_offsets(col_off_C, row_off_C, part_n_C, part_m_C, m_whole, n_whole, M_TILDE, N_TILDE);

    for ( dim_t r = 0; r < FMM_BLIS_MULTS; r++ )
    {

        paramsA.nsplit = 0;
        paramsB.nsplit = 0;
        paramsC.nsplit = 0;

        for (dim_t isplits = 0; isplits < M_TILDE * K_TILDE; isplits++)
        {
            ((float*)paramsA.coef)[paramsA.nsplit] = _U(isplits, r);
            paramsA.off_m[paramsA.nsplit] = row_off_A[isplits];
            paramsA.off_n[paramsA.nsplit] = col_off_A[isplits];
            paramsA.part_m[paramsA.nsplit] = part_m_A[isplits];
            paramsA.part_n[paramsA.nsplit] = part_n_A[isplits];
            paramsA.nsplit++;
        }

        for (dim_t isplits = 0; isplits < K_TILDE * N_TILDE; isplits++)
        {
            ((float*)paramsB.coef)[paramsB.nsplit] = _V(isplits, r);
            paramsB.off_m[paramsB.nsplit] = row_off_B[isplits];
            paramsB.off_n[paramsB.nsplit] = col_off_B[isplits];
            paramsB.part_m[paramsB.nsplit] = part_m_B[isplits];
            paramsB.part_n[paramsB.nsplit] = part_n_B[isplits];
            paramsB.nsplit++;
        }

        for (dim_t isplits = 0; isplits < M_TILDE * N_TILDE; isplits++)
        {
            ((float*)paramsC.coef)[paramsC.nsplit] = _W(isplits, r);
            paramsC.off_m[paramsC.nsplit] = row_off_C[isplits];
            paramsC.off_n[paramsC.nsplit] = col_off_C[isplits];
            paramsC.part_m[paramsC.nsplit] = part_m_C[isplits];
            paramsC.part_n[paramsC.nsplit] = part_n_C[isplits];
            paramsC.nsplit++;
        }

        bli_l3_thread_decorator
        (
            &A_local,
            &B_local,
            &C_local,
            cntx,
            ( cntl_t* )&cntl,
            rntm
        );
    }
}

void bli_strassen_ab( obj_t* alpha, obj_t* A, obj_t* B, obj_t* beta, obj_t* C )
{
    bli_strassen_ab_ex( alpha, A, B, beta, C, STRASSEN_FMM );
}