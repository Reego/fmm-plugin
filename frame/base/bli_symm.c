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
    }
}

void bli_strassen_ab_symm( obj_t* alpha, obj_t* A, obj_t* B, obj_t* beta, obj_t* C) {
    bli_strassen_ab_symm_ex(alpha, A, B, beta, C, CLASSICAL_FMM);
    // bli_strassen_ab_symm_ex(alpha, A, B, beta, C, STRASSEN_FMM);
}