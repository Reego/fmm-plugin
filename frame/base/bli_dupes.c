#include "blis.h"
#include "bli_fmm.h"

void bli_packm_blk_var1_dupe
     (
       const obj_t*     c,
             obj_t*     p,
       const cntx_t*    cntx,
       const cntl_t*    cntl,
             thrinfo_t* thread
     )
{
	// Extract various fields from the control tree.
	double start_time = 0.0;
	if (TIME_PACK) {
		start_time = _bl_clock();
	}
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
	packm_ker_ft packm_ker_cast = bli_packm_def_cntl_ukr( cntl );
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
	if (TIME_PACK) {
		double end_time = _bl_clock();
		if (ldp == 8) {
			TIMES[2] += end_time - start_time;
			CLOCK_CALLS[2] += 1;
		}
		else {
			TIMES[3] += end_time - start_time;
			CLOCK_CALLS[3] += 1;
		}
	}
}

typedef void (*xpbys_mxn_ft)
    (
            dim_t m,
            dim_t n,
      const void* x, inc_t rs_x, inc_t cs_x,
      const void* b,
            void* y, inc_t rs_y, inc_t cs_y
    );

#undef  GENTFUNC2
#define GENTFUNC2(ctypex,ctypey,chx,chy,op) \
\
BLIS_INLINE void PASTEMAC2(chx,chy,op) \
    ( \
            dim_t m, \
            dim_t n, \
      const void* x, inc_t rs_x, inc_t cs_x, \
      const void* b, \
            void* y, inc_t rs_y, inc_t cs_y \
    ) \
{ \
	const ctypex* restrict x_cast = x; \
	const ctypey* restrict b_cast = b; \
	      ctypey* restrict y_cast = y; \
\
	PASTEMAC3(chx,chy,chy,xpbys_mxn) \
	( \
	  m, n, \
	  x_cast, rs_x, cs_x, \
	  b_cast, \
	  y_cast, rs_y,  cs_y \
	); \
}

INSERT_GENTFUNC2_BASIC(xpbys_mxn_fn);
INSERT_GENTFUNC2_MIX_DP(xpbys_mxn_fn);

static xpbys_mxn_ft GENARRAY2_ALL(xpbys_mxn, xpbys_mxn_fn);

void bli_gemm_ker_var2_dupe
     (
       const obj_t*     a,
       const obj_t*     b,
       const obj_t*     c,
       const cntx_t*    cntx,
       const cntl_t*    cntl,
             thrinfo_t* thread_par
     )
{
	double start_time = 0.0;
	if (TIME_C) {
		start_time = _bl_clock();
	}

	      num_t  dt_exec   = bli_obj_exec_dt( c );
	      num_t  dt_c      = bli_obj_dt( c );

	const pack_t schema_a  = bli_obj_pack_schema( a );
	const pack_t schema_b  = bli_obj_pack_schema( b );

	      dim_t  m         = bli_obj_length( c );
	      dim_t  n         = bli_obj_width( c );
	      dim_t  k         = bli_obj_width( a );

	const char*  a_cast    = bli_obj_buffer_at_off( a );
	const inc_t  is_a      = bli_obj_imag_stride( a );
	      dim_t  pd_a      = bli_obj_panel_dim( a );
	      inc_t  ps_a      = bli_obj_panel_stride( a );

	const char*  b_cast    = bli_obj_buffer_at_off( b );
	const inc_t  is_b      = bli_obj_imag_stride( b );
	      dim_t  pd_b      = bli_obj_panel_dim( b );
	      inc_t  ps_b      = bli_obj_panel_stride( b );

	      char*  c_cast    = bli_obj_buffer_at_off( c );
	      inc_t  rs_c      = bli_obj_row_stride( c );
	      inc_t  cs_c      = bli_obj_col_stride( c );

	// If any dimension is zero, return immediately.
	if ( bli_zero_dim3( m, n, k ) ) return;

	// Detach and multiply the scalars attached to A and B.
	// NOTE: We know that the internal scalars of A and B are already of the
	// target datatypes because the necessary typecasting would have already
	// taken place during bli_packm_init().
	obj_t scalar_a, scalar_b;
	bli_obj_scalar_detach( a, &scalar_a );
	bli_obj_scalar_detach( b, &scalar_b );
	bli_mulsc( &scalar_a, &scalar_b );

	// Grab the addresses of the internal scalar buffers for the scalar
	// merged above and the scalar attached to C.
	// NOTE: We know that scalar_b is of type dt_exec due to the above code
	// that casts the scalars of A and B to dt_exec via scalar_a and scalar_b,
	// and we know that the internal scalar in C is already of the type dt_c
	// due to the casting in the implementation of bli_obj_scalar_attach().
	const char* alpha_cast = bli_obj_internal_scalar_buffer( &scalar_b );
	const char* beta_cast  = bli_obj_internal_scalar_buffer( c );

	/*
#ifdef BLIS_ENABLE_GEMM_MD
	// Tweak parameters in select mixed domain cases (rcc, crc, ccr).
	if ( bli_cntx_method( cntx ) == BLIS_NAT )
	{
		bli_gemm_md_ker_var2_recast
		(
		  &dt_exec,
		  bli_obj_dt( a ),
		  bli_obj_dt( b ),
		  &dt_c,
		  &m, &n, &k,
		  &pd_a, &ps_a,
		  &pd_b, &ps_b,
		  c,
		  &rs_c, &cs_c
		);
	}
#endif
	*/

	const siz_t dt_size   = bli_dt_size( dt_exec );
	const siz_t dt_c_size = bli_dt_size( dt_c );

	// Alias some constants to simpler names.
	const dim_t MR = pd_a;
	const dim_t NR = pd_b;

	// Query the context for the micro-kernel address and cast it to its
	// function pointer type.
	gemm_ukr_ft gemm_ukr = bli_gemm_var_cntl_ukr( cntl );
	const void* params   = bli_gemm_var_cntl_params( cntl );

	// Temporary C buffer for edge cases. Note that the strides of this
	// temporary buffer are set so that they match the storage of the
	// original C matrix. For example, if C is column-stored, ct will be
	// column-stored as well.
	char        ct[ BLIS_STACK_BUF_MAX_SIZE ]
	                __attribute__((aligned(BLIS_STACK_BUF_ALIGN_SIZE)));
	const bool  row_pref    = bli_gemm_var_cntl_row_pref( cntl );
	const inc_t rs_ct       = ( row_pref ? NR : 1 );
	const inc_t cs_ct       = ( row_pref ? 1 : MR );
	const char* zero        = bli_obj_buffer_for_const( dt_exec, &BLIS_ZERO );

	//
	// Assumptions/assertions:
	//   rs_a == 1
	//   cs_a == PACKMR
	//   pd_a == MR
	//   ps_a == stride to next micro-panel of A
	//   rs_b == PACKNR
	//   cs_b == 1
	//   pd_b == NR
	//   ps_b == stride to next micro-panel of B
	//   rs_c == (no assumptions)
	//   cs_c == (no assumptions)
	//

	// Compute number of primary and leftover components of the m and n
	// dimensions.
	const dim_t n_iter = n / NR + ( n % NR ? 1 : 0 );
	const dim_t n_left = n % NR;

	const dim_t m_iter = m / MR + ( m % MR ? 1 : 0 );
	const dim_t m_left = m % MR;

	// Determine some increments used to step through A, B, and C.
	const inc_t rstep_a = ps_a * dt_size;

	const inc_t cstep_b = ps_b * dt_size;

	const inc_t rstep_c = rs_c * MR * dt_c_size;
	const inc_t cstep_c = cs_c * NR * dt_c_size;

	auxinfo_t aux;

	// Save the pack schemas of A and B to the auxinfo_t object.
	bli_auxinfo_set_schema_a( schema_a, &aux );
	bli_auxinfo_set_schema_b( schema_b, &aux );

	// Save the imaginary stride of A and B to the auxinfo_t object.
	bli_auxinfo_set_is_a( is_a, &aux );
	bli_auxinfo_set_is_b( is_b, &aux );

	// Save the virtual microkernel address and the params.
	bli_auxinfo_set_ukr( gemm_ukr, &aux );
	bli_auxinfo_set_params( params, &aux );

	dim_t jr_start, jr_end, jr_inc;
	dim_t ir_start, ir_end, ir_inc;

#ifdef BLIS_ENABLE_JRIR_TLB

	// Query the number of threads and thread ids for the jr loop around
	// the microkernel.
	thrinfo_t* thread = bli_thrinfo_sub_node( 0, thread_par );
	const dim_t jr_nt  = bli_thrinfo_n_way( thread );
	const dim_t jr_tid = bli_thrinfo_work_id( thread );

	const dim_t ir_nt  = 1;
	const dim_t ir_tid = 0;

	dim_t n_ut_for_me
	=
	bli_thread_range_tlb_d( jr_nt, jr_tid, m_iter, n_iter, MR, NR,
	                        &jr_start, &ir_start );

	// Always increment by 1 in both dimensions.
	jr_inc = 1;
	ir_inc = 1;

	// Each thread iterates over the entire panel of C until it exhausts its
	// assigned set of microtiles.
	jr_end = n_iter;
	ir_end = m_iter;

	// Successive iterations of the ir loop should start at 0.
	const dim_t ir_next = 0;

#else // ifdef ( _SLAB || _RR )

	// Query the number of threads and thread ids for the ir loop around
	// the microkernel.
	thrinfo_t* thread = bli_thrinfo_sub_node( 0, thread_par );
	thrinfo_t* caucus = bli_thrinfo_sub_node( 0, thread );
	const dim_t jr_nt  = bli_thrinfo_n_way( thread );
	const dim_t jr_tid = bli_thrinfo_work_id( thread );
	const dim_t ir_nt  = bli_thrinfo_n_way( caucus );
	const dim_t ir_tid = bli_thrinfo_work_id( caucus );

	// Determine the thread range and increment for the 2nd and 1st loops.
	// NOTE: The definition of bli_thread_range_slrr() will depend on whether
	// slab or round-robin partitioning was requested at configure-time.
	bli_thread_range_slrr( jr_tid, jr_nt, n_iter, 1, FALSE, &jr_start, &jr_end, &jr_inc );
	bli_thread_range_slrr( ir_tid, ir_nt, m_iter, 1, FALSE, &ir_start, &ir_end, &ir_inc );

	// Calculate the total number of microtiles assigned to this thread.
	dim_t n_ut_for_me = ( ( ir_end + ir_inc - 1 - ir_start ) / ir_inc ) *
	                    ( ( jr_end + jr_inc - 1 - jr_start ) / jr_inc );

	// Each succesive iteration of the ir loop always starts at ir_start.
	const dim_t ir_next = ir_start;

#endif

	// It's possible that there are so few microtiles relative to the number
	// of threads that one or more threads gets no work. If that happens, those
	// threads can return early.
	if ( n_ut_for_me == 0 ) return;

	// Loop over the n dimension (NR columns at a time).
	for ( dim_t j = jr_start; j < jr_end; j += jr_inc )
	{
		const char* b1 = b_cast + j * cstep_b;
		      char* c1 = c_cast + j * cstep_c;

		// Compute the current microtile's width.
		const dim_t n_cur = ( bli_is_not_edge_f( j, n_iter, n_left )
		                      ? NR : n_left );

		// Initialize our next panel of B to be the current panel of B.
		const char* b2 = b1;

		// Loop over the m dimension (MR rows at a time).
		for ( dim_t i = ir_start; i < ir_end; i += ir_inc )
		{
			const char* a1  = a_cast + i * rstep_a;
			      char* c11 = c1     + i * rstep_c;

			// Compute the current microtile's length.
			const dim_t m_cur = ( bli_is_not_edge_f( i, m_iter, m_left )
			                      ? MR : m_left );

			// Compute the addresses of the next panels of A and B.
			const char* a2 = bli_gemm_get_next_a_upanel( a1, rstep_a, ir_inc );
			if ( bli_is_last_iter_slrr( i, ir_end, ir_tid, ir_nt ) )
			{
				a2 = a_cast;
				b2 = bli_gemm_get_next_b_upanel( b1, cstep_b, jr_inc );
			}

			// printf("\n\nGEMM KER VAR 2 \t offset %d %d \t of %d and %d\n\n", i*MR, j*NR, m, n);

			// Save addresses of next panels of A and B to the auxinfo_t
			// object.
			bli_auxinfo_set_next_a( a2, &aux );
			bli_auxinfo_set_next_b( b2, &aux );

			// TODO

			// bli_auxinfo_set_off_m( i*MR + off_mc, &aux );
			// bli_auxinfo_set_off_n( j*NR + off_nc, &aux );

			// Edge case handling now occurs within the microkernel itself, but
			// we must still explicitly accumulate to a temporary microtile in
			// situations where a virtual microkernel is being used, such as
			// during the 1m method or some cases of mixed datatypes.
			if ( dt_exec == dt_c )
			{
				// Invoke the gemm micro-kernel.
				gemm_ukr
				(
				  m_cur,
				  n_cur,
				  k,
				  ( void* )alpha_cast,
				  ( void* )a1,
				  ( void* )b1,
				  ( void* )beta_cast,
				           c11, rs_c, cs_c,
				  &aux,
				  ( cntx_t* )cntx
				);
			}
			else
			{
				// Invoke the gemm micro-kernel.
				gemm_ukr
				(
				  MR,
				  NR,
				  k,
				  ( void* )alpha_cast,
				  ( void* )a1,
				  ( void* )b1,
				  ( void* )zero,
				           &ct, rs_ct, cs_ct,
				  &aux,
				  ( cntx_t* )cntx
				);

				// Accumulate to C with typecasting.
				xpbys_mxn[ dt_exec ][ dt_c ]
				(
				  m_cur, n_cur,
				  &ct, rs_ct, cs_ct,
				  ( void* )beta_cast,
				  c11, rs_c, cs_c
				);
			}

			// Decrement the number of microtiles assigned to the thread; once
			// it reaches zero, return immediately.
			n_ut_for_me -= 1; if ( n_ut_for_me == 0 ) break;
		}
		ir_start = ir_next;
	}
	if (TIME_C) {
		double end_time = _bl_clock();
		TIMES[1] += end_time - start_time;
		CLOCK_CALLS[1] += 1;
	}
}