#include "blis.h"
#include "bli_fmm.h"

#define __U( i,j ) fmm->U[ (i)*fmm->R + (j) ]
#define __V( i,j ) fmm->V[ (i)*fmm->R + (j) ]
#define __W( i,j ) fmm->W[ (i)*fmm->R + (j) ]

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

    fmm_cntl_t* fmm_cntl = (fmm_cntl_t*) cntl;
    fmm_t* fmm = fmm_cntl->fmm;

    gemm_cntl_t* gemm_cntl = fmm_cntl->gemm_cntl;

    fmm_gemm_cntl_t* fmm_gemm_cntl = (fmm_gemm_cntl_t*) cntl;

    fmm_params_t paramsA;
    fmm_params_t paramsB;
    fmm_params_t paramsC;

    // TODO
    // dim_t n = bli_obj_length( c );
    // dim_t m = bli_obj_width( c );
    // dim_t k = bli_obj_width( a );
    dim_t m = bli_obj_length( c );
    dim_t n = bli_obj_width( c );
    dim_t k = bli_obj_width( a );

    dim_t m_edge, m_whole, k_edge, k_whole, n_edge, n_whole;

    bl_fmm_acquire_spart (fmm->m_tilde, fmm->k_tilde, 0, 0, a, &a0 );
    bl_fmm_acquire_spart (fmm->k_tilde, fmm->n_tilde, 0, 0, b, &b0 );
    bl_fmm_acquire_spart (fmm->m_tilde, fmm->n_tilde, 0, 0, c, &c0 );

    bli_obj_alias_submatrix( &a0, &a_local );
    bli_obj_alias_submatrix( &b0, &b_local );
    bli_obj_alias_submatrix( &c0, &c_local );

    paramsA.m_max = m; paramsA.n_max = k;
    paramsB.m_max = n; paramsB.n_max = k; // B gets transposed
    paramsC.m_max = m; paramsC.n_max = n;
    paramsC.local = &c_local;

    func_t *pack_ukr;

    // bli_gemm_cntl_set_packa_params((const void *) &paramsB, gemm_cntl);
    // bli_gemm_cntl_set_packb_params((const void *) &paramsA, gemm_cntl);
    bli_gemm_cntl_set_packa_params((const void *) &paramsA, gemm_cntl);
    bli_gemm_cntl_set_packb_params((const void *) &paramsB, gemm_cntl);
    bli_gemm_cntl_set_params((const void *) &paramsC, gemm_cntl);

    dim_t row_off_A[fmm->m_tilde * fmm->k_tilde], col_off_A[fmm->m_tilde * fmm->k_tilde];
    dim_t part_m_A[fmm->m_tilde * fmm->k_tilde], part_n_A[fmm->m_tilde * fmm->k_tilde];

    init_part_offsets(row_off_A, col_off_A, part_m_A, part_n_A, m, k, fmm->m_tilde, fmm->k_tilde);

    dim_t row_off_B[fmm->k_tilde * fmm->n_tilde], col_off_B[fmm->k_tilde * fmm->n_tilde];
    dim_t part_m_B[fmm->k_tilde * fmm->n_tilde], part_n_B[fmm->k_tilde * fmm->n_tilde];

    // need to stop order of the col_off, row_off and part_n and part_m pairs because
    // B gets transposed before packing.
    init_part_offsets(col_off_B, row_off_B, part_n_B, part_m_B, k, n, fmm->k_tilde, fmm->n_tilde); // since B is transposed... something idk.

    dim_t row_off_C[fmm->m_tilde * fmm->n_tilde], col_off_C[fmm->m_tilde * fmm->n_tilde];
    dim_t part_m_C[fmm->m_tilde * fmm->n_tilde], part_n_C[fmm->m_tilde * fmm->n_tilde];

    init_part_offsets(row_off_C, col_off_C, part_m_C, part_n_C, m, n, fmm->m_tilde, fmm->n_tilde);

    if (1)
    {
        paramsA.reindex = false;
        paramsB.reindex = false;
    }
    else
    {
        // TODO
        // generate partition matrices

        paramsA.reindex = true;
        paramsB.reindex = true;

        paramsA.parts = (obj_t*) malloc(fmm->m_tilde * fmm->k_tilde * sizeof(obj_t));
        paramsB.parts = (obj_t*) malloc(fmm->k_tilde * fmm->n_tilde * sizeof(obj_t));

        // obj_t atemp_local;

        // // We always pass B^T to bli_l3_packm.
        // bli_obj_alias_to( a, &atemp_local );
        // if ( bli_obj_has_trans( a ) )
        // {
        //     bli_obj_set_onlytrans( BLIS_NO_TRANSPOSE, &atemp_local );
        // }
        // else
        // {
        //     bli_obj_induce_trans( &atemp_local );
        // }

        for (int i = 0; i < fmm->m_tilde; i++)
        {
            for (int j = 0; j < fmm->k_tilde; j++)
            {
                int part_index = i * fmm->k_tilde + j;

                dim_t partm = part_m_A[part_index];
                dim_t partn = part_n_A[part_index];

                inc_t offm = row_off_A[part_index];
                inc_t offn = col_off_A[part_index];

                obj_t temp;

                bli_acquire_mpart(
                    offm,
                    offn,
                    partm,
                    partn,
                    a,
                    &temp
                );

            //     bli_obj_alias_to( a, &a0 );
            // bli_obj_alias_to( b, &b0 );
            // bli_obj_alias_to( c, &c0 );
                bli_obj_create( bli_obj_dt(&temp), partm, partn, 0, 0, &(paramsA.parts[part_index]) );
                bli_copym(&temp, &(paramsA.parts[part_index]));

                // printf("\n\n\t%ld %ld - %d %d\n", temp.cs, temp.rs, paramsA.parts[part_index].cs, paramsA.parts[part_index].rs);
                // bli_printm( "\tmatrix 'atemp_local', initialized by columns:", &atemp_local, "%5.3f", "" );
            }
        }

        obj_t btemp_local;

        // We always pass B^T to bli_l3_packm.
        bli_obj_alias_to( b, &btemp_local );
        bli_obj_induce_trans( &btemp_local );

        for (int i = 0; i < fmm->k_tilde; i++)
        {
            for (int j = 0; j < fmm->n_tilde; j++)
            {
                int part_index = i * fmm->k_tilde + j;

                dim_t partm = part_m_B[part_index];
                dim_t partn = part_n_B[part_index];

                inc_t offm = row_off_B[part_index];
                inc_t offn = col_off_B[part_index];

                obj_t temp;

                bli_acquire_mpart(
                    offm,
                    offn,
                    partm,
                    partn,
                    &btemp_local,
                    &temp
                );

                // bli_obj_alias_submatrix(&temp, &(paramsB.parts[part_index]));
                // bli_copym(&temp, &(paramsB.parts[part_index]));

                bli_obj_create( bli_obj_dt(&temp), partm, partn, 0, 0, &(paramsB.parts[part_index]) );
                bli_copym(&temp, &(paramsB.parts[part_index]));

                // printf("\n\n\t%ld %ld - %d %d\n", temp.cs, temp.rs, paramsB.parts[part_index].cs, paramsB.parts[part_index].rs);
                // bli_printm( "\tmatrix 'temp', initialized by columns:", &temp, "%5.3f", "" );
                // bli_printm( "\tmatrix 'btemp_local', initialized by columns:", &btemp_local, "%5.3f", "" );
            
            }
        }
    }



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

    if(paramsA.reindex)
    {
        for (int i = 0; i < fmm->m_tilde * fmm->k_tilde; i++)
        {
            bli_obj_free(&(paramsA.parts[i]));
        }
        free(paramsA.parts);
    }

    if (paramsB.reindex)
    {
        for (int i = 0; i < fmm->k_tilde * fmm->n_tilde; i++)
        {
            bli_obj_free(&(paramsB.parts[i]));
        }
        free(paramsB.parts);
    }
}