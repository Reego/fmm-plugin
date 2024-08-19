#include "blis.h"
#include "bli_fmm.h"

#define __U( i,j ) fmm->U[ (i)*fmm->R + (j) ]
#define __V( i,j ) fmm->V[ (i)*fmm->R + (j) ]
#define __W( i,j ) fmm->W[ (i)*fmm->R + (j) ]

void bli_packm_blk_var1_fmm_static
     (
       const obj_t*     c,
             obj_t*     p,
       const cntx_t*    cntx,
       const cntl_t*    cntl,
             thrinfo_t* thread,
             packm_ker_ft packm_ker_cast
     )
{
    // Extract various fields from the control tree.
    pack_t schema  = bli_packm_def_cntl_pack_schema( cntl );
    bool   invdiag = bli_packm_def_cntl_does_invert_diag( cntl );
    bool   revifup = bli_packm_def_cntl_rev_iter_if_upper( cntl );
    bool   reviflo = bli_packm_def_cntl_rev_iter_if_lower( cntl );

    // Every thread initializes p and determines the size of memory block
    // needed (which gets embedded into the otherwise "blank" mem_t entry
    // in the control tree node). Return early if no packing is required.
    // If the requested size is zero, then we don't need to do any allocation.
    siz_t size_p = bli_packm_init( c, p, cntl );
    if ( size_p == 0 )
        return;

    // Update the buffer address in p to point to the buffer associated
    // with the mem_t entry acquired from the memory broker (now cached in
    // the control tree node).
    void* buffer = bli_packm_alloc( size_p, cntl, thread );
    bli_obj_set_buffer( buffer, p );

    // Check parameters.
    if ( bli_error_checking_is_enabled() )
        bli_packm_int_check( c, p );

    num_t   dt_c           = bli_obj_dt( c );
    dim_t   dt_c_size      = bli_dt_size( dt_c );

    num_t   dt_p           = bli_obj_dt( p );
    dim_t   dt_p_size      = bli_dt_size( dt_p );

    struc_t strucc         = bli_obj_struc( c );
    doff_t  diagoffc       = bli_obj_diag_offset( c );
    diag_t  diagc          = bli_obj_diag( c );
    uplo_t  uploc          = bli_obj_uplo( c );
    conj_t  conjc          = bli_obj_conj_status( c );

    dim_t   iter_dim       = bli_obj_length( p );
    dim_t   panel_len_full = bli_obj_width( p );
    dim_t   panel_len_max  = bli_obj_padded_width( p );

    char*   c_cast         = bli_obj_buffer_at_off( c );
    inc_t   incc           = bli_obj_row_stride( c );
    inc_t   ldc            = bli_obj_col_stride( c );
    dim_t   panel_dim_off  = bli_obj_row_off( c );
    dim_t   panel_len_off  = bli_obj_col_off( c );

    char*   p_cast         = bli_obj_buffer( p );
    inc_t   ldp            = bli_obj_col_stride( p );
    dim_t   panel_dim_max  = bli_obj_panel_dim( p );
    inc_t   ps_p           = bli_obj_panel_stride( p );
    dim_t   bcast_p        = bli_packm_def_cntl_bmult_m_bcast( cntl );

    doff_t  diagoffc_inc   = ( doff_t )panel_dim_max;

    obj_t   kappa_local;
    char*   kappa_cast     = bli_packm_scalar( &kappa_local, p );

    // Query the datatype-specific function pointer from the control tree.
    const void*  params         = bli_packm_def_cntl_ukr_params( cntl );

    // Compute the total number of iterations we'll need.
    dim_t n_iter = iter_dim / panel_dim_max + ( iter_dim % panel_dim_max ? 1 : 0 );

    // Set the initial values and increments for indices related to C and P
    // based on whether reverse iteration was requested.
    dim_t  ic0, ip0;
    doff_t ic_inc, ip_inc;

    if ( ( revifup && bli_is_upper( uploc ) && bli_is_triangular( strucc ) ) ||
         ( reviflo && bli_is_lower( uploc ) && bli_is_triangular( strucc ) ) )
    {
        ic0    = (n_iter - 1) * panel_dim_max;
        ic_inc = -panel_dim_max;
        ip0    = n_iter - 1;
        ip_inc = -1;
    }
    else
    {
        ic0    = 0;
        ic_inc = panel_dim_max;
        ip0    = 0;
        ip_inc = 1;
    }

    // Query the number of threads (single-member thread teams) and the thread
    // team ids from the current thread's packm thrinfo_t node.
    const dim_t nt  = bli_thrinfo_num_threads( thread );
    const dim_t tid = bli_thrinfo_thread_id( thread );

    // Determine the thread range and increment using the current thread's
    // packm thrinfo_t node. NOTE: The definition of bli_thread_range_slrr()
    // will depend on whether slab or round-robin partitioning was requested
    // at configure-time.
    dim_t it_start, it_end, it_inc;
    bli_thread_range_slrr( tid, nt, n_iter, 1, FALSE, &it_start, &it_end, &it_inc );

    char* p_begin = p_cast;

    if ( !bli_is_triangular( strucc ) ||
         bli_is_stored_subpart_n( diagoffc, uploc, iter_dim, panel_len_full ) )
    {
        // This case executes if the panel is either dense, belongs
        // to a Hermitian or symmetric matrix, which includes stored,
        // unstored, and diagonal-intersecting panels, or belongs
        // to a completely stored part of a triangular matrix.

        // Iterate over every logical micropanel in the source matrix.
        for ( dim_t ic  = ic0,    ip  = ip0,    it  = 0; it < n_iter;
                    ic += ic_inc, ip += ip_inc, it += 1 )
        {
            dim_t  panel_dim_i     = bli_min( panel_dim_max, iter_dim - ic );
            dim_t  panel_dim_off_i = panel_dim_off + ic;

            char*  c_begin         = c_cast   + (ic  )*incc*dt_c_size;

            // Hermitian/symmetric and general packing may use slab or round-
            // robin (bli_is_my_iter()), depending on which was selected at
            // configure-time.
            if ( bli_is_my_iter( it, it_start, it_end, tid, nt ) )
            {
                packm_ker_cast
                (
                  bli_is_triangular( strucc ) ? BLIS_GENERAL : strucc,
                  diagc,
                  uploc,
                  conjc,
                  schema,
                  invdiag,
                  panel_dim_i,
                  panel_len_full,
                  panel_dim_max,
                  panel_len_max,
                  panel_dim_off_i,
                  panel_len_off,
                  bcast_p,
                  kappa_cast,
                  c_begin, incc, ldc,
                  p_begin,       ldp,
                  params,
                  cntx
                );
            }

            p_begin += ps_p*dt_p_size;
        }
    }
    else
    {
        // This case executes if the panel belongs to a diagonal-intersecting
        // part of a triangular matrix.

        // Iterate over every logical micropanel in the source matrix.
        for ( dim_t ic  = ic0,    ip  = ip0,    it  = 0; it < n_iter;
                    ic += ic_inc, ip += ip_inc, it += 1 )
        {
            dim_t  panel_dim_i     = bli_min( panel_dim_max, iter_dim - ic );
            dim_t  panel_dim_off_i = panel_dim_off + ic;

            doff_t diagoffc_i      = diagoffc + (ip  )*diagoffc_inc;
            char*  c_begin         = c_cast   + (ic  )*incc*dt_c_size;

            if ( bli_is_unstored_subpart_n( diagoffc_i, uploc, panel_dim_i,
                                            panel_len_full ) )
                continue;

            // Sanity check. Diagonals should not intersect the short edge of
            // a micro-panel (typically corresponding to a register blocksize).
            // If they do, then the constraints on cache blocksizes being a
            // whole multiple of the register blocksizes was somehow violated.
            if ( ( diagoffc_i > -panel_dim_i &&
                   diagoffc_i < 0 ) ||
                 ( diagoffc_i > panel_len_full &&
                   diagoffc_i < panel_len_full + panel_dim_i ) )
                bli_check_error_code( BLIS_NOT_YET_IMPLEMENTED );

            dim_t panel_off_i     = 0;
            dim_t panel_len_i     = panel_len_full;
            dim_t panel_len_max_i = panel_len_max;

            if ( bli_intersects_diag_n( diagoffc_i, panel_dim_i, panel_len_full ) )
            {
                if ( bli_is_lower( uploc ) )
                {
                    panel_off_i     = 0;
                    panel_len_i     = diagoffc_i + panel_dim_i;
                    panel_len_max_i = bli_min( diagoffc_i + panel_dim_max,
                                               panel_len_max );
                }
                else // if ( bli_is_upper( uploc ) )
                {
                    panel_off_i     = diagoffc_i;
                    panel_len_i     = panel_len_full - panel_off_i;
                    panel_len_max_i = panel_len_max  - panel_off_i;
                }
            }

            dim_t panel_len_off_i = panel_off_i + panel_len_off;

            char* c_use           = c_begin + (panel_off_i  )*ldc*dt_c_size;
            char* p_use           = p_begin;

            // We need to re-compute the imaginary stride as a function of
            // panel_len_max_i since triangular packed matrices have panels
            // of varying lengths. NOTE: This imaginary stride value is
            // only referenced by the packm kernels for induced methods.
            inc_t is_p_use = ldp * panel_len_max_i;

            // We nudge the imaginary stride up by one if it is odd.
            is_p_use += ( bli_is_odd( is_p_use ) ? 1 : 0 );

            // NOTE: We MUST use round-robin work allocation (bli_is_my_iter_rr())
            // when packing micropanels of a triangular matrix.
            if ( bli_is_my_iter_rr( it, tid, nt ) )
            {
                packm_ker_cast
                (
                  strucc,
                  diagc,
                  uploc,
                  conjc,
                  schema,
                  invdiag,
                  panel_dim_i,
                  panel_len_i,
                  panel_dim_max,
                  panel_len_max_i,
                  panel_dim_off_i,
                  panel_len_off_i,
                  bcast_p,
                  kappa_cast,
                  c_use, incc, ldc,
                  p_use,       ldp,
                  params,
                  cntx
                );
            }

            // NOTE: This value is usually LESS than ps_p because triangular
            // matrices usually have several micro-panels that are shorter
            // than a "full" micro-panel.
            p_begin += is_p_use*dt_p_size;
        }
    }
}


// void bli_l3_packa_fmm <-- can be found in old commits

void bli_l3_packa_fmm_static
     (
       const obj_t*  a,
       const obj_t*  b,
       const obj_t*  c,
       const cntx_t* cntx,
       const cntl_t* cntl,
             thrinfo_t* thread_par
     )
{
	obj_t a_local, a_pack;

	bli_obj_alias_to( a, &a_local );
	if ( bli_obj_has_trans( a ) )
	{
		bli_obj_induce_trans( &a_local );
		bli_obj_set_onlytrans( BLIS_NO_TRANSPOSE, &a_local );
	}

	// Pack matrix A according to the control tree node.

	fmm_params_t* paramsA = (fmm_params_t*) bli_packm_def_cntl_ukr_params( cntl );

	packm_ker_ft packm_kers[] = {
		// dSTRASSEN_PACK_A_0,
		// dSTRASSEN_PACK_A_1,
		// dSTRASSEN_PACK_A_2,
		// dSTRASSEN_PACK_A_3,
		// dSTRASSEN_PACK_A_4,
		// dSTRASSEN_PACK_A_5,
		// dSTRASSEN_PACK_A_6,
  //       dSTRASSEN_PACK_A_7
        // dCLASSICAL_PACK_A_0,
        // dCLASSICAL_PACK_A_1,
        // dCLASSICAL_PACK_A_2,
        // dCLASSICAL_PACK_A_3,
        // dCLASSICAL_PACK_A_4,
        // dCLASSICAL_PACK_A_5,
        // dCLASSICAL_PACK_A_6,
        // dCLASSICAL_PACK_A_7
        dFMM_222_PACK_A_0,
        dFMM_222_PACK_A_1,
        dFMM_222_PACK_A_2,
        dFMM_222_PACK_A_3,
        dFMM_222_PACK_A_4,
        dFMM_222_PACK_A_5,
        dFMM_222_PACK_A_6
	};

	packm_ker_ft packm_ker_cast = packm_kers[paramsA->r];

	{ // bli_packm_int

		// Barrier so that we know threads are done with previous computation
		// with the same packing buffer before starting to pack.
		thrinfo_t* thread = bli_thrinfo_sub_node( 0, thread_par );
		bli_thrinfo_barrier( thread );

		bli_packm_blk_var1_fmm_static
		(
		  &a_local,
	  	  &a_pack,
		  cntx,
		  cntl,
		  thread,
		  packm_ker_cast
		);

		// Barrier so that packing is done before computation.
		bli_thrinfo_barrier( thread );
	}

	// Proceed with execution using packed matrix A.
	bli_l3_int
	(
	  &a_pack,
	  b,
	  c,
	  cntx,
	  bli_cntl_sub_node( 0, cntl ),
	  bli_thrinfo_sub_node( 0, thread_par )
	);
}

void bli_l3_packb_fmm_static
     (
       const obj_t*  a,
       const obj_t*  b,
       const obj_t*  c,
       const cntx_t* cntx,
       const cntl_t* cntl,
             thrinfo_t* thread_par
     )
{

	obj_t bt_local, bt_pack;

	// We always pass B^T to bli_l3_packm.
	bli_obj_alias_to( b, &bt_local );
	if ( bli_obj_has_trans( b ) )
	{
		bli_obj_set_onlytrans( BLIS_NO_TRANSPOSE, &bt_local );
	}
	else
	{
		bli_obj_induce_trans( &bt_local );
	}

	fmm_params_t* paramsB = (fmm_params_t*) bli_packm_def_cntl_ukr_params( cntl );

	packm_ker_ft packm_kers[] = {
        // dCLASSICAL_PACK_B_0,
        // dCLASSICAL_PACK_B_1,
        // dCLASSICAL_PACK_B_2,
        // dCLASSICAL_PACK_B_3,
        // dCLASSICAL_PACK_B_4,
        // dCLASSICAL_PACK_B_5,
        // dCLASSICAL_PACK_B_6,
        // dCLASSICAL_PACK_B_7
		dFMM_222_PACK_B_0,
        dFMM_222_PACK_B_1,
        dFMM_222_PACK_B_2,
        dFMM_222_PACK_B_3,
        dFMM_222_PACK_B_4,
        dFMM_222_PACK_B_5,
        dFMM_222_PACK_B_6
	};

	packm_ker_ft packm_ker_cast = packm_kers[paramsB->r];


	{ // bli_packm_int

		// Barrier so that we know threads are done with previous computation
		// with the same packing buffer before starting to pack.
		thrinfo_t* thread = bli_thrinfo_sub_node( 0, thread_par );
		bli_thrinfo_barrier( thread );

		bli_packm_blk_var1_fmm_static
		(
		  &bt_local,
		  &bt_pack,
		  cntx,
		  cntl,
		  thread,
		  packm_ker_cast
		);

		// Barrier so that packing is done before computation.
		bli_thrinfo_barrier( thread );
	}

	// Transpose packed object back to B.
	bli_obj_induce_trans( &bt_pack );

	// Proceed with execution using packed matrix B.
	bli_l3_int
	(
	  a,
	  &bt_pack,
	  c,
	  cntx,
	  bli_cntl_sub_node( 0, cntl ),
	  bli_thrinfo_sub_node( 0, thread_par )
	);
}