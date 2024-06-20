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

/* Classical blocked matrix multiplication */
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


void bli_fmm_cntl_init_pushb
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
             int var
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
             int var
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

    row_left = m % row_splits;
    col_left = n % col_splits;

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

void bli_strassen_ab_ex_var( obj_t* alpha, obj_t* A, obj_t* B, obj_t* beta, obj_t* C, fmm_t* fmm, int variant) {

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

    obj_t A0, B0, C0;

    bli_obj_alias_submatrix( A, &A_local );
    bli_obj_alias_submatrix( B, &B_local );
    bli_obj_alias_submatrix( C, &C_local );

    gemm_cntl_t cntl0;
    gemm_cntl_t* cntl;

    if (variant == -1) {

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

        fmm_cntl->fmm = fmm;
        fmm_cntl->gemm_cntl = cntl;

        bli_gemm_cntl_set_packa_params((const void *) fmm, cntl);
        bli_gemm_cntl_set_packb_params((const void *) fmm, cntl);
        bli_gemm_cntl_set_params((const void *) fmm, cntl);

        bli_l3_thread_decorator
        (
            &A_local,
            &B_local,
            &C_local,
            cntx,
            ( cntl_t* )&fmm_gemm_cntl,
            rntm
        );
        return;
    }

    // with reordering variant

    fmm_gemm_cntl_alt_t fmm_gemm_cntl;
    fmm_cntl_t* fmm_cntl = &(fmm_gemm_cntl.fmm_cntl);

    cntl = &fmm_gemm_cntl.gemm_cntl;

    if (variant == 2) {
        bli_fmm_cntl_init_pushb
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
          fmm_cntl,
          variant
        );
    }
    else {
        bli_fmm_gemm_cntl_init_var
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
          fmm_cntl,
          variant
        );
    }

    func_t *pack_ukr;

    pack_ukr = bli_cntx_get_ukrs( FMM_BLIS_PACK_UKR, cntx );
    bli_gemm_cntl_set_packa_ukr_simple( pack_ukr , cntl );
    bli_gemm_cntl_set_packb_ukr_simple( bli_cntx_get_ukrs( FMM_BLIS_PACK_UKR, cntx ), cntl );
    bli_gemm_cntl_set_ukr_simple( bli_cntx_get_ukrs( FMM_BLIS_GEMM_UKR, cntx ), cntl );

    fmm_cntl->fmm = fmm;
    fmm_cntl->gemm_cntl = cntl;

    if (variant == 2) {
        bli_gemm_cntl_set_packa_params((const void *) &fmm_gemm_cntl, cntl);
        bli_gemm_cntl_set_packb_params((const void *) &fmm_gemm_cntl, cntl);
        bli_gemm_cntl_set_params((const void *) &fmm_gemm_cntl, cntl);
    }
    else {
        bli_gemm_cntl_set_packa_params((const void *) fmm, cntl);
        bli_gemm_cntl_set_packb_params((const void *) fmm, cntl);
        bli_gemm_cntl_set_params((const void *) fmm, cntl);
    }

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

void bli_strassen_ab_ex( obj_t* alpha, obj_t* A, obj_t* B, obj_t* beta, obj_t* C, fmm_t* fmm ) {
    bli_strassen_ab_ex_var(alpha, A, B, beta, C, fmm, 0);
}

void bli_strassen_ab( obj_t* alpha, obj_t* A, obj_t* B, obj_t* beta, obj_t* C )
{
    bli_strassen_ab_ex( alpha, A, B, beta, C, &STRASSEN_FMM );
}