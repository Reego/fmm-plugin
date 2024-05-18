#include "blis.h"
#include "bli_fmm.h"
#include <time.h>
#include <complex.h>

#define _U( i,j ) fmm.U[ (i)*fmm.R + (j) ]
#define _V( i,j ) fmm.V[ (i)*fmm.R + (j) ]
#define _W( i,j ) fmm.W[ (i)*fmm.R + (j) ]

#define __U( i,j ) fmm->U[ (i)*fmm->R + (j) ]
#define __V( i,j ) fmm->V[ (i)*fmm->R + (j) ]
#define __W( i,j ) fmm->W[ (i)*fmm->R + (j) ]


static packm_ker_ft GENARRAY(packm_struc_cxk,packm_struc_cxk);
static packm_ker_ft GENARRAY2_ALL(packm_struc_cxk_md,packm_struc_cxk_md);


/* Define Strassen's algorithm */
int STRASSEN_FMM_U[4][7] = {{1, 0, 1, 0, 1, -1, 0}, {0, 0, 0, 0, 1, 0, 1}, {0, 1, 0, 0, 0, 1, 0}, {1, 1, 0, 1, 0, 0, -1}};
int STRASSEN_FMM_V[4][7] = {{1, 1, 0, -1, 0, 1, 0}, {0, 0, 1, 0, 0, 1, 0}, {0, 0, 0, 1, 0, 0, 1}, {1, 0, -1, 0, 1, 0, 1}};
int STRASSEN_FMM_W[4][7] = {{1, 0, 0, 1, -1, 0, 1}, {0, 0, 1, 0, 1, 0, 0}, {0, 1, 0, 1, 0, 0, 0}, {1, -1, 1, 0, 0, 1, 0}};

int CLASSICAL_FMM_U[4][8] = {{1, 0, 1, 0, 0, 0, 0, 0}, {0, 1, 0, 1, 0, 0, 0, 0}, {0, 0, 0, 0, 1, 0, 1, 0}, {0, 0, 0, 0, 0, 1, 0, 1}};
int CLASSICAL_FMM_V[4][8] = {{1, 0, 0, 0, 1, 0, 0, 0}, {0, 0, 1, 0, 0, 0, 1, 0}, {0, 1, 0, 0, 0, 1, 0, 0}, {0, 0, 0, 1, 0, 0, 0, 1}};
int CLASSICAL_FMM_W[4][8] = {{1, 1, 0, 0, 0, 0, 0, 0}, {0, 0, 1, 1, 0, 0, 0, 0}, {0, 0, 0, 0, 1, 1, 0, 0}, {0, 0, 0, 0, 0, 0, 1, 1}};

fmm_t STRASSEN_FMM = {
    .m_tilde = 2,
    .n_tilde = 2,
    .k_tilde = 2,
    .R = 7,
    .U = &STRASSEN_FMM_U,
    .V = &STRASSEN_FMM_V,
    .W = &STRASSEN_FMM_W,
};

fmm_t CLASSICAL_FMM = {
    .m_tilde = 2,
    .n_tilde = 2,
    .k_tilde = 2,
    .R = 8,
    .U = &CLASSICAL_FMM_U,
    .V = &CLASSICAL_FMM_V,
    .W = &CLASSICAL_FMM_W,
};


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
    // if ( bli_error_checking_is_enabled() )
    //  bli_gemm_check( alpha, A, B, beta, C, cntx );

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

    obj_t A0, B0, C0;

    bli_obj_alias_submatrix( A, &A_local );
    bli_obj_alias_submatrix( B, &B_local );
    bli_obj_alias_submatrix( C, &C_local );

    gemm_cntl_t cntl0;
    gemm_cntl_t* cntl;

    fmm_gemm_cntl_t fmm_gemm_cntl;
    fmm_cntl_t* fmm_cntl = &(fmm_gemm_cntl.fmm_cntl);

    cntl = &fmm_gemm_cntl.gemm_cntl;
    bli_fmm_gemm_cntl_init
    (
      im,
      BLIS_GEMM,
      alpha,
      &A_local,
      &B_local,
      beta,
      &C_local,
      cntx,
      cntl,
      fmm_cntl
    );

    func_t *pack_ukr;

    pack_ukr = bli_cntx_get_ukrs( FMM_BLIS_PACK_UKR, cntx );
    bli_gemm_cntl_set_packa_ukr_simple( pack_ukr , cntl );
    bli_gemm_cntl_set_packb_ukr_simple( bli_cntx_get_ukrs( FMM_BLIS_PACK_UKR, cntx ), cntl );
    bli_gemm_cntl_set_ukr_simple( bli_cntx_get_ukrs( FMM_BLIS_GEMM_UKR, cntx ), cntl );

    fmm_cntl->fmm = &fmm;

    bli_gemm_cntl_set_packa_params((const void *) &fmm, cntl);
    bli_gemm_cntl_set_packb_params((const void *) &fmm, cntl);
    bli_gemm_cntl_set_params((const void *) &fmm, cntl);

    bli_l3_thread_decorator
    (
        &A_local,
        &B_local,
        &C_local,
        cntx,
        ( cntl_t* )&fmm_gemm_cntl,
        rntm
    );
}

void bli_strassen_ab( obj_t* alpha, obj_t* A, obj_t* B, obj_t* beta, obj_t* C )
{
    bli_strassen_ab_ex( alpha, A, B, beta, C, CLASSICAL_FMM );
    // bli_strassen_ab_ex( alpha, A, B, beta, C, STRASSEN_FMM );
}

void bli_strassen_ab_symm_ex( obj_t* alpha, obj_t* A, obj_t* B, obj_t* beta, obj_t* C, fmm_t fmm) {

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
    //  bli_gemm_check( alpha, A, B, beta, C, cntx );

    // Check for zero dimensions, alpha == 0, or other conditions which
    // mean that we don't actually have to perform a full l3 operation.
    if ( bli_l3_return_early_if_trivial( alpha, A, B, beta, C ) == BLIS_SUCCESS )
        return;

    // Default to using native execution.
    num_t dt = bli_obj_dt( C );
    ind_t im = BLIS_NAT;

    // if ( bli_obj_is_complex( C ) )
    // {
    //     printf("NLOOOOOO WAY\n\n\n");
    //     // Find the highest priority induced method that is both enabled and
    //     // available for the current operation. (If an induced method is
    //     // available but not enabled, or simply unavailable, BLIS_NAT will
    //     // be returned here.)
    //     im = bli_symmind_find_avail( dt );
    //     const prec_t comp_prec = bli_obj_comp_prec( C );
    //     const num_t dt_comp = ( im == BLIS_1M ? BLIS_REAL : bli_dt_domain( dt ) ) | comp_prec;
    //     // im = bli_gemmind_find_avail( dt );
    // }

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

#if 1
    bli_obj_alias_submatrix( &A0, &A_local );
    bli_obj_alias_submatrix( &B0, &B_local );
    bli_obj_alias_submatrix( &C0, &C_local );

    // if (1) {
    //     A_local = a;
    //     B_local = b;
    //     C_local = c;
    // }
#else
    bli_obj_alias_submatrix( A, &A_local );
    bli_obj_alias_submatrix( B, &B_local );
    bli_obj_alias_submatrix( C, &C_local );
#endif
    gemm_cntl_t cntl0;
    gemm_cntl_t* cntl;

    fmm_gemm_cntl_t fmm_gemm_cntl;
    // printf("&fmm_gemm_cntl %p    &(fmm_gemm_cntl.fmm_cntl) %p\n\n", &fmm_gemm_cntl, &(fmm_gemm_cntl.fmm_cntl));
    fmm_cntl_t* fmm_cntl = &(fmm_gemm_cntl.fmm_cntl);
    // printf("\n%p\n", fmm_cntl);

    if (0) {
        cntl = &fmm_gemm_cntl.gemm_cntl;
        bli_fmm_gemm_cntl_init
        (
          im,
          BLIS_GEMM,
          alpha,
          A,
          B,
          beta,
          C,
          cntx,
          cntl,
          fmm_cntl
        );
        // printf("RAN bli_fmm_gemm_cntl_init\n");
    }
    else {
        cntl = &cntl0;
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
          cntl
        );
    }

    fmm_params_t paramsA, paramsB, paramsC;

    paramsA.m_max = m; paramsA.n_max = k;
    paramsB.m_max = n; paramsB.n_max = k;
    paramsC.m_max = n; paramsC.n_max = m;
    paramsC.local = &C_local;

#if 1
    func_t *pack_ukr;

    // printf("Checkpoint A\n\n");

    pack_ukr = bli_cntx_get_ukrs( FMM_BLIS_PACK_UKR_SYMM, cntx );
    bli_gemm_cntl_set_packa_ukr_simple( pack_ukr , cntl );
    bli_gemm_cntl_set_packb_ukr_simple( bli_cntx_get_ukrs( FMM_BLIS_PACK_UKR_SYMM, cntx ), cntl );
    bli_gemm_cntl_set_ukr_simple( bli_cntx_get_ukrs( FMM_BLIS_GEMM_UKR, cntx ), cntl );

    // printf("Checkpoint B\n\n");

    bli_gemm_cntl_set_packa_params((const void *) &paramsB, cntl);
    bli_gemm_cntl_set_packb_params((const void *) &paramsA, cntl);
    bli_gemm_cntl_set_params((const void *) &paramsC, cntl);

    // printf("Checkpoint C\n\n\n");

    // handle complex values
    if ( im == BLIS_1M )
    {
        printf("\t\t!!!!\n\n");
        // printf("Checkpoint D\n\n");
        gemm_ukr_ft gemm_ukr      = bli_cntx_get_ukr_dt( dt, BLIS_GEMM_UKR, cntx );

        bli_gemm_var_cntl_set_real_ukr_simple(gemm_ukr, cntl);
        bli_gemm_var_cntl_set_ukr_simple(
            bli_cntx_get_ukr_dt(dt, FMM_BLIS_GEMM1M_UKR, cntx), cntl
        );
    }

    ////
    if (0) {

        // printf("Wait, really? %p %p\n\n", fmm_cntl, &(fmm_cntl->fmm));

        fmm_cntl->fmm = &fmm;

        bli_gemm_cntl_set_packa_params((const void *) &fmm, cntl);
        bli_gemm_cntl_set_packb_params((const void *) &fmm, cntl);
        bli_gemm_cntl_set_params((const void *) &fmm, cntl);

        // printf("About to call bli_l3_thread_decorator\n\n");

        // bli_gemm_cntl_set_params((const void *) fmm, &cntl);
        bli_l3_thread_decorator
        (
            A,
            B,
            C,
            cntx,
            ( cntl_t* )&fmm_gemm_cntl,
            rntm
        );
        return;
    }
    ////

    bli_gemm_cntl_set_packa_params((const void *) &paramsB, cntl);
    bli_gemm_cntl_set_packb_params((const void *) &paramsA, cntl);
    bli_gemm_cntl_set_params((const void *) &paramsC, cntl);
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

    for ( dim_t r = 0; r < fmm.R; r++ )
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
            ( cntl_t* )cntl,
            rntm
        );

        if (0) return; // TODO
    }
}

void bli_strassen_ab_symm( obj_t* alpha, obj_t* A, obj_t* B, obj_t* beta, obj_t* C) {
    bli_strassen_ab_symm_ex(alpha, A, B, beta, C, CLASSICAL_FMM);
    // bli_strassen_ab_symm_ex(alpha, A, B, beta, C, STRASSEN_FMM);
}























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
     )
{

    const prec_t      comp_prec     = bli_obj_comp_prec( c );
    const num_t       dt_c          = bli_obj_dt( c );
    const num_t       dt_comp       = ( im == BLIS_1M ? BLIS_REAL : bli_dt_domain( dt_c ) ) | comp_prec;
          gemm_ukr_ft gemm_ukr      = bli_cntx_get_ukr_dt( dt_comp, BLIS_GEMM_UKR, cntx );
          gemm_ukr_ft real_gemm_ukr = NULL;
    const bool        row_pref      = bli_cntx_get_ukr_prefs_dt( dt_comp, BLIS_GEMM_UKR_ROW_PREF, cntx );

    // An optimization: If C is stored by rows and the micro-kernel prefers
    // contiguous columns, or if C is stored by columns and the micro-kernel
    // prefers contiguous rows, transpose the entire operation to allow the
    // micro-kernel to access elements of C in its preferred manner.
    bool needs_swap = (   row_pref && bli_obj_is_col_tilted( c ) ) ||
                      ( ! row_pref && bli_obj_is_row_tilted( c ) );

    // NOTE: This case casts right-side symm/hemm/trmm/trmm3 in terms of left side.
    // This may be necessary when the current subconfiguration uses a gemm microkernel
    // that assumes that the packing kernel will have already duplicated
    // (broadcast) element of B in the packed copy of B. Supporting
    // duplication within the logic that packs micropanels from symmetric
    // matrices is ugly, but technically supported. This can
    // lead to the microkernel being executed on an output matrix with the
    // microkernel's general stride IO case (unless the microkernel supports
    // both both row and column IO cases as well). As a
    // consequence, those subconfigurations need a way to force the symmetric
    // matrix to be on the left (and thus the general matrix to the on the
    // right). So our solution is that in those cases, the subconfigurations
    // simply #define BLIS_DISABLE_{SYMM,HEMM,TRMM,TRMM3}_RIGHT.

    // If A is being multiplied from the right, transpose all operands
    // so that we can perform the computation as if A were being multiplied
    // from the left.
#ifdef BLIS_DISABLE_SYMM_RIGHT
    if ( family == BLIS_SYMM )
        needs_swap = bli_obj_is_symmetric( b );
#endif
#ifdef BLIS_DISABLE_HEMM_RIGHT
    if ( family == BLIS_HEMM )
        needs_swap = bli_obj_is_hermitian( b );
#endif
#ifdef BLIS_DISABLE_TRMM_RIGHT
    if ( family == BLIS_TRMM )
        needs_swap = bli_obj_is_triangular( b );
#endif
#ifdef BLIS_DISABLE_TRMM3_RIGHT
    if ( family == BLIS_TRMM3 )
        needs_swap = bli_obj_is_triangular( b );
#endif

    // Swap the A and B operands if required. This transforms the operation
    // C = alpha A B + beta C into C^T = alpha B^T A^T + beta C^T.
    if ( needs_swap )
    {
        bli_obj_swap( a, b );

        bli_obj_induce_trans( a );
        bli_obj_induce_trans( b );
        bli_obj_induce_trans( c );
    }

    // If alpha is non-unit, typecast and apply it to the scalar attached
    // to B, unless it happens to be triangular.
    if ( bli_obj_root_is_triangular( b ) )
    {
        if ( !bli_obj_equals( alpha, &BLIS_ONE ) )
            bli_obj_scalar_apply_scalar( alpha, a );
    }
    else // if ( bli_obj_root_is_triangular( b ) )
    {
        if ( !bli_obj_equals( alpha, &BLIS_ONE ) )
            bli_obj_scalar_apply_scalar( alpha, b );
    }

    // If beta is non-unit, typecast and apply it to the scalar attached
    // to C.
    if ( !bli_obj_equals( beta, &BLIS_ONE ) )
        bli_obj_scalar_apply_scalar( beta, c );

    void_fp macro_kernel_fp = family == BLIS_GEMM ||
                              family == BLIS_HEMM ||
                              family == BLIS_SYMM ? bli_gemm_ker_var2 :
#ifdef BLIS_ENABLE_JRIR_TLB
                              family == BLIS_GEMMT ?
                                 bli_obj_is_lower( c ) ? bli_gemmt_l_ker_var2b : bli_gemmt_u_ker_var2b :
                              family == BLIS_TRMM ||
                              family == BLIS_TRMM3 ?
                                  bli_obj_is_triangular( a ) ?
                                     bli_obj_is_lower( a ) ? bli_trmm_ll_ker_var2b : bli_trmm_lu_ker_var2b :
                                     bli_obj_is_lower( b ) ? bli_trmm_rl_ker_var2b : bli_trmm_ru_ker_var2b :
                              NULL; // Should never happen
#else
                              family == BLIS_GEMMT ?
                                 bli_obj_is_lower( c ) ? bli_gemmt_l_ker_var2 : bli_gemmt_u_ker_var2 :
                              family == BLIS_TRMM ||
                              family == BLIS_TRMM3 ?
                                  bli_obj_is_triangular( a ) ?
                                     bli_obj_is_lower( a ) ? bli_trmm_ll_ker_var2 : bli_trmm_lu_ker_var2 :
                                     bli_obj_is_lower( b ) ? bli_trmm_rl_ker_var2 : bli_trmm_ru_ker_var2 :
                              NULL; // Should never happen
#endif

    const num_t         dt_a          = bli_obj_dt( a );
    const num_t         dt_b          = bli_obj_dt( b );
    const num_t         dt_ap         = bli_dt_domain( dt_a ) | comp_prec;
    const num_t         dt_bp         = bli_dt_domain( dt_b ) | comp_prec;
    const bool          trmm_r        = family == BLIS_TRMM && bli_obj_is_triangular( b );
    const bool          a_lo_tri      = bli_obj_is_triangular( a ) && bli_obj_is_lower( a );
    const bool          b_up_tri      = bli_obj_is_triangular( b ) && bli_obj_is_upper( b );
          pack_t        schema_a      = BLIS_PACKED_ROW_PANELS;
          pack_t        schema_b      = BLIS_PACKED_COL_PANELS;
    const packm_ker_ft  packm_a_ukr   = dt_a == dt_ap ? packm_struc_cxk[ dt_a ]
                                                      : packm_struc_cxk_md[ dt_a ][ dt_ap ];
    const packm_ker_ft  packm_b_ukr   = dt_b == dt_bp ? packm_struc_cxk[ dt_b ]
                                                      : packm_struc_cxk_md[ dt_b ][ dt_bp ];
    const dim_t         mr_def        = bli_cntx_get_blksz_def_dt( dt_comp, BLIS_MR, cntx );
    const dim_t         mr_pack       = bli_cntx_get_blksz_max_dt( dt_comp, BLIS_MR, cntx );
    const dim_t         mr_bcast      = bli_cntx_get_blksz_max_dt( dt_comp, BLIS_BBM, cntx );
          dim_t         mr_scale      = 1;
    const dim_t         nr_def        = bli_cntx_get_blksz_def_dt( dt_comp, BLIS_NR, cntx );
    const dim_t         nr_pack       = bli_cntx_get_blksz_max_dt( dt_comp, BLIS_NR, cntx );
    const dim_t         nr_bcast      = bli_cntx_get_blksz_max_dt( dt_comp, BLIS_BBN, cntx );
          dim_t         nr_scale      = 1;
    const dim_t         kr_def        = bli_cntx_get_blksz_def_dt( dt_comp, BLIS_KR, cntx );
    const dim_t         mc_def        = bli_cntx_get_blksz_def_dt( dt_comp, BLIS_MC, cntx );
    const dim_t         mc_max        = bli_cntx_get_blksz_max_dt( dt_comp, BLIS_MC, cntx );
          dim_t         mc_scale      = 1;
    const dim_t         nc_def        = bli_cntx_get_blksz_def_dt( dt_comp, BLIS_NC, cntx );
    const dim_t         nc_max        = bli_cntx_get_blksz_max_dt( dt_comp, BLIS_NC, cntx );
          dim_t         nc_scale      = 1;
    const dim_t         kc_def        = bli_cntx_get_blksz_def_dt( dt_comp, BLIS_KC, cntx );
    const dim_t         kc_max        = bli_cntx_get_blksz_max_dt( dt_comp, BLIS_KC, cntx );
          dim_t         kc_scale      = 1;

    if ( im == BLIS_1M )
    {
        printf("HEYYYYY\n\n\n\n\n");
        if ( ! row_pref )
        {
            schema_a = BLIS_PACKED_ROW_PANELS_1E;
            schema_b = BLIS_PACKED_COL_PANELS_1R;
            mr_scale = 2;
            mc_scale = 2;
        }
        else
        {
            schema_a = BLIS_PACKED_ROW_PANELS_1R;
            schema_b = BLIS_PACKED_COL_PANELS_1E;
            nr_scale = 2;
            nc_scale = 2;
        }

        kc_scale = 2;
        real_gemm_ukr = gemm_ukr;
        gemm_ukr = bli_cntx_get_ukr_dt( dt_comp, BLIS_GEMM1M_UKR, cntx );
    }

#if 0
#ifdef BLIS_ENABLE_GEMM_MD
    cntx_t cntx_local;

    // If any of the storage datatypes differ, or if the computation precision
    // differs from the storage precision of C, utilize the mixed datatype
    // code path.
    // NOTE: If we ever want to support the caller setting the computation
    // domain explicitly, we will need to check the computation dt against the
    // storage dt of C (instead of the computation precision against the
    // storage precision of C).
    if ( bli_obj_dt( &c_local ) != bli_obj_dt( &a_local ) ||
         bli_obj_dt( &c_local ) != bli_obj_dt( &b_local ) ||
         bli_obj_comp_prec( &c_local ) != bli_obj_prec( &c_local ) )
    {
        // Handle mixed datatype cases in bli_gemm_md(), which may modify
        // the objects or the context. (If the context is modified, cntx
        // is adjusted to point to cntx_local.)
        bli_gemm_md( &a_local, &b_local, beta, &c_local, &schema_a, &schema_b, &cntx_local, &cntx );
    }
#endif
#endif

    // Create two nodes for the macro-kernel.
    bli_cntl_init_node
    (
      NULL,         // variant function pointer not used
      &cntl->ir_loop
    );

    bli_gemm_var_cntl_init_node
    (
      macro_kernel_fp,
      dt_comp,
      dt_c,
      gemm_ukr,
      real_gemm_ukr,
      row_pref,
      mr_def / mr_scale,
      nr_def / nr_scale,
      mr_scale,
      nr_scale,
      &cntl->ker
    );
    bli_cntl_attach_sub_node
    (
      BLIS_THREAD_NR,
      ( cntl_t* )&cntl->ir_loop,
      ( cntl_t* )&cntl->ker
    );

    // Give the gemm kernel control tree node to the
    // virtual microkernel as the parameters, so that e.g.
    // the 1m virtual microkernel can look up the real-domain
    // micro-kernel and its parameters.
    bli_gemm_var_cntl_set_params( &cntl->ker, ( cntl_t* )&cntl->ker );

    // Create a node for packing matrix A.
    bli_packm_def_cntl_init_node
    (
      bli_l3_packa, // pack the left-hand operand
      dt_a,
      dt_ap,
      dt_comp,
      packm_a_ukr,
      mr_def / mr_scale,
      mr_pack,
      mr_bcast,
      mr_scale,
      kr_def,
      FALSE,
      FALSE,
      FALSE,
      schema_a,
      BLIS_BUFFER_FOR_A_BLOCK,
      &cntl->pack_a
    );
    bli_cntl_attach_sub_node
    (
      BLIS_THREAD_NONE,
      ( cntl_t* )&cntl->ker,
      ( cntl_t* )&cntl->pack_a
    );

    // Create a node for partitioning the m dimension by MC.
    bli_part_cntl_init_node
    (
      bli_gemm_blk_var1,
      dt_comp,
      mc_def / mc_scale,
      mc_max / mc_scale,
      mc_scale,
      mr_def / mr_scale,
      mr_scale,
      a_lo_tri ? BLIS_BWD : BLIS_FWD,
      bli_obj_is_triangular( a ) || bli_obj_is_upper_or_lower( c ),
      &cntl->part_ic
    );
    bli_cntl_attach_sub_node
    (
      trmm_r ? BLIS_THREAD_MC | BLIS_THREAD_NC : BLIS_THREAD_MC,
      ( cntl_t* )&cntl->pack_a,
      ( cntl_t* )&cntl->part_ic
    );

    // Create a node for packing matrix B.
    bli_packm_def_cntl_init_node
    (
      bli_l3_packb, // pack the right-hand operand
      dt_b,
      dt_bp,
      dt_comp,
      packm_b_ukr,
      nr_def / nr_scale,
      nr_pack,
      nr_bcast,
      nr_scale,
      kr_def,
      FALSE,
      FALSE,
      FALSE,
      schema_b,
      BLIS_BUFFER_FOR_B_PANEL,
      &cntl->pack_b
    );
    bli_cntl_attach_sub_node
    (
      BLIS_THREAD_NONE,
      ( cntl_t* )&cntl->part_ic,
      ( cntl_t* )&cntl->pack_b
    );

    // Create a node for partitioning the k dimension by KC.
    bli_part_cntl_init_node
    (
      bli_gemm_blk_var3,
      dt_comp,
      kc_def / kc_scale,
      kc_max / kc_scale,
      kc_scale,
      kr_def,
      1,
      a_lo_tri || b_up_tri ? BLIS_BWD : BLIS_FWD,
      FALSE,
      &cntl->part_pc
    );
    bli_cntl_attach_sub_node
    (
      BLIS_THREAD_KC,
      ( cntl_t* )&cntl->pack_b,
      ( cntl_t* )&cntl->part_pc
    );

    // Create a node for partitioning the n dimension by NC.
    bli_part_cntl_init_node
    (
      bli_gemm_blk_var2,
      dt_comp,
      nc_def / nc_scale,
      nc_max / nc_scale,
      nc_scale,
      nr_def / nr_scale,
      nr_scale,
      b_up_tri ? BLIS_BWD : BLIS_FWD,
      bli_obj_is_triangular( b ) || bli_obj_is_upper_or_lower( c ),
      &cntl->part_jc
    );
    bli_cntl_attach_sub_node
    (
      trmm_r ? BLIS_THREAD_NONE : BLIS_THREAD_NC,
      ( cntl_t* )&cntl->part_pc,
      ( cntl_t* )&cntl->part_jc
    );


    // Create a node for FMM partitioning
    bli_cntl_init_node
    (
      bli_fmm_cntl,
      fmm_cntl
    );
    bli_cntl_attach_sub_node
    (
      trmm_r ? BLIS_THREAD_NONE : BLIS_THREAD_NC,
      ( cntl_t* )&cntl->part_jc,
      ( cntl_t* )fmm_cntl
    );

    bli_gemm_cntl_finalize
    (
      family,
      a,
      b,
      c,
      cntl
    );
}




void bl_fmm_acquire_spart 
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

    bli_obj_set_padded_dims( m, n, sub_obj );

    offm_inc = split_rowidx * row_part;
    offn_inc = split_colidx * col_part;

    //Taken directly from BLIS. Need to verify if this is still true. 
    // Compute the diagonal offset based on the m and n offsets.
    doff_t diagoff_inc = ( doff_t )offm_inc - ( doff_t )offn_inc;

    bli_obj_inc_offs( offm_inc, offn_inc, sub_obj );
    bli_obj_inc_diag_offset( diagoff_inc, sub_obj );

}

void bli_fmm_cntl
     (
       const obj_t*     a,
       const obj_t*     b,
       const obj_t*     c,
       const cntx_t*    cntx,
       const cntl_t*    cntl,
             thrinfo_t* thread_par
     )
{   

    thrinfo_t* thread = bli_thrinfo_sub_node( 0, thread_par );

    obj_t a0, b0, c0;
    obj_t a_local, b_local, c_local;

    bli_obj_alias_to( a, &a0 );
    bli_obj_alias_to( b, &b0 );
    bli_obj_alias_to( c, &c0 );

    fmm_gemm_cntl_t* fmm_gemm_cntl = (fmm_gemm_cntl_t*) cntl;

    gemm_cntl_t* gemm_cntl = &fmm_gemm_cntl->gemm_cntl;

    fmm_t* fmm = (fmm_t*) bli_gemm_cntl_params(&fmm_gemm_cntl->gemm_cntl);

    fmm_params_t paramsA;
    fmm_params_t paramsB;
    fmm_params_t paramsC;

    // TODO
    dim_t n = bli_obj_length( c );
    dim_t m = bli_obj_width( c );
    dim_t k = bli_obj_width( a );

    dim_t m_edge, m_whole, k_edge, k_whole, n_edge, n_whole;

    bl_fmm_acquire_spart (fmm->m_tilde, fmm->k_tilde, 0, 0, a, &a0 );
    bl_fmm_acquire_spart (fmm->k_tilde, fmm->n_tilde, 0, 0, b, &b0 );
    bl_fmm_acquire_spart (fmm->k_tilde, fmm->n_tilde, 0, 0, c, &c0 );

    bli_obj_alias_submatrix( &a0, &a_local );
    bli_obj_alias_submatrix( &b0, &b_local );
    bli_obj_alias_submatrix( &c0, &c_local );

    paramsA.m_max = m; paramsA.n_max = k;
    paramsB.m_max = n; paramsB.n_max = k;
    paramsC.m_max = n; paramsC.n_max = m;
    paramsC.local = &c_local;

    func_t *pack_ukr;

    bli_gemm_cntl_set_packa_params((const void *) &paramsB, gemm_cntl);
    bli_gemm_cntl_set_packb_params((const void *) &paramsA, gemm_cntl);
    bli_gemm_cntl_set_params((const void *) &paramsC, gemm_cntl);

    dim_t row_off_A[fmm->m_tilde * fmm->k_tilde], col_off_A[fmm->m_tilde * fmm->k_tilde];
    dim_t part_m_A[fmm->m_tilde * fmm->k_tilde], part_n_A[fmm->m_tilde * fmm->k_tilde];

    init_part_offsets(row_off_A, col_off_A, part_m_A, part_n_A, m, k, fmm->m_tilde, fmm->k_tilde);

    dim_t row_off_B[fmm->k_tilde * fmm->n_tilde], col_off_B[fmm->k_tilde * fmm->n_tilde];
    dim_t part_m_B[fmm->k_tilde * fmm->n_tilde], part_n_B[fmm->k_tilde * fmm->n_tilde];

    init_part_offsets(col_off_B, row_off_B, part_n_B, part_m_B, k, n, fmm->k_tilde, fmm->n_tilde); // since B is transposed... something idk.

    dim_t row_off_C[fmm->m_tilde * fmm->n_tilde], col_off_C[fmm->m_tilde * fmm->n_tilde];
    dim_t part_m_C[fmm->m_tilde * fmm->n_tilde], part_n_C[fmm->m_tilde * fmm->n_tilde];

    init_part_offsets(col_off_C, row_off_C, part_n_C, part_m_C, m, n, fmm->m_tilde, fmm->n_tilde);

    for ( dim_t r = 0; r < fmm->R; r++ )
    {
        paramsA.nsplit = 0;
        paramsB.nsplit = 0;
        paramsC.nsplit = 0;

        for (dim_t isplits = 0; isplits < fmm->m_tilde * fmm->k_tilde; isplits++)
        {
            ((float*)paramsA.coef)[paramsA.nsplit] = __U(isplits, r);
            paramsA.off_m[paramsA.nsplit] = row_off_A[isplits];
            paramsA.off_n[paramsA.nsplit] = col_off_A[isplits];
            paramsA.part_m[paramsA.nsplit] = part_m_A[isplits];
            paramsA.part_n[paramsA.nsplit] = part_n_A[isplits];
            paramsA.nsplit++;
        }

        for (dim_t isplits = 0; isplits < fmm->k_tilde * fmm->n_tilde; isplits++)
        {
            ((float*)paramsB.coef)[paramsB.nsplit] = __V(isplits, r);
            paramsB.off_m[paramsB.nsplit] = row_off_B[isplits];
            paramsB.off_n[paramsB.nsplit] = col_off_B[isplits];
            paramsB.part_m[paramsB.nsplit] = part_m_B[isplits];
            paramsB.part_n[paramsB.nsplit] = part_n_B[isplits];
            paramsB.nsplit++;
        }

        for (dim_t isplits = 0; isplits < fmm->m_tilde * fmm->n_tilde; isplits++)
        {
            ((float*)paramsC.coef)[paramsC.nsplit] = __W(isplits, r);
            paramsC.off_m[paramsC.nsplit] = row_off_C[isplits];
            paramsC.off_n[paramsC.nsplit] = col_off_C[isplits];
            paramsC.part_m[paramsC.nsplit] = part_m_C[isplits];
            paramsC.part_n[paramsC.nsplit] = part_n_C[isplits];
            paramsC.nsplit++;
        }

        bli_l3_int
        (
          &a_local,
          &b_local,
          &c_local,
          cntx,
          bli_cntl_sub_node( 0, cntl ),
          thread
        );
    }

    bli_gemm_cntl_set_params((void*) &fmm, cntl);
}