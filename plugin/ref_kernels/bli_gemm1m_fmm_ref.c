#include "blis.h"
#include STRINGIFY_INT(../PASTEMAC(plugin,BLIS_PNAME_INFIX).h)

#undef  GENTFUNCRO
#define GENTFUNCRO( ctype_r, ctype, chr, ch, opname, arch, suf ) \
\
void PASTEMAC3(chr,opname,arch,suf) \
     ( \
             dim_t      m, \
             dim_t      n, \
             dim_t      k, \
       const void*      alpha0, \
       const void*      a0, \
       const void*      b0, \
       const void*      beta0, \
             void*      c0, inc_t rs_c, inc_t cs_c, \
             auxinfo_t* auxinfo, \
       const cntx_t*    cntx  \
     ) \
{ \
	const ctype*      alpha     = alpha0; \
	const ctype*      a         = a0; \
	const ctype*      b         = b0; \
	const ctype*      beta      = beta0; \
	      ctype*      c         = c0; \
\
	const num_t       dt       = PASTEMAC(ch,type); \
	const gemm_ukr_ft ukr      = bli_cntx_get_ukr_dt( dt, BLIS_GEMM_UKR, cntx ); \
	const bool        row_pref = bli_cntx_get_ukr_prefs_dt( dt, BLIS_GEMM_UKR_ROW_PREF, cntx ); \
	const dim_t       mr       = bli_cntx_get_blksz_def_dt( dt, BLIS_MR, cntx ); \
	const dim_t       nr       = bli_cntx_get_blksz_def_dt( dt, BLIS_NR, cntx ); \
\
	const gemm_ukr_ft rgemm_ukr = bli_cntx_get_ukr_dt( dt, BLIS_GEMM_UKR, cntx ); /*bli_gemm_var_cntl_real_ukr( params );*/ \
	/*const void*       params_r  = bli_gemm_var_cntl_real_params( params );*/ \
\
	const dim_t       mr_r      = row_pref ? mr : 2 * mr; \
	const dim_t       nr_r      = row_pref ? 2 * nr : nr; \
\
	/* Convert the micro-tile dimensions from being in units of complex elements to*/ \
	/* be in units of real elements. */ \
	const dim_t       m_r        = row_pref ? m : 2 * m; \
	const dim_t       n_r        = row_pref ? 2 * n : n; \
	const dim_t       k_r        = 2 * k; \
\
	      ctype       ct[ BLIS_STACK_BUF_MAX_SIZE / sizeof( ctype ) ] \
	                  __attribute__((aligned(BLIS_STACK_BUF_ALIGN_SIZE))); \
	      inc_t       rs_ct = row_pref ? n_r : 1;\
	      inc_t       cs_ct = row_pref ? 1 : m_r;\
\
	const ctype_r* restrict a_r     = ( const ctype_r* ) a; \
\
	const ctype_r* restrict b_r     = ( const ctype_r* ) b; \
\
	const ctype_r* restrict one_r   = PASTEMAC(chr,1); \
	const ctype_r* restrict zero_r  = PASTEMAC(chr,0); \
\
	const ctype_r* restrict alpha_r = &PASTEMAC(ch,real)( *alpha ); \
	const ctype_r* restrict alpha_i = &PASTEMAC(ch,imag)( *alpha ); \
\
	const ctype_r* restrict beta_r  = &PASTEMAC(ch,real)( *beta ); \
	const ctype_r* restrict beta_i  = &PASTEMAC(ch,imag)( *beta ); \
\
	      ctype_r*          c_use; \
	      inc_t             rs_c_use; \
	      inc_t             cs_c_use; \
\
	      ctype       ab[ BLIS_STACK_BUF_MAX_SIZE / sizeof(ctype_r) ]; \
	const ctype*      zero     = PASTEMAC(ch,0); \
\
	fmm_params_t* params = ( fmm_params_t* )bli_auxinfo_params( auxinfo ); \
	dim_t nsplit = params->nsplit; \
	float* restrict coef = ( float* )params->coef; \
	inc_t* restrict off_m = params->off_m; \
	inc_t* restrict off_n = params->off_n; \
	inc_t* restrict part_m = params->part_m; \
	inc_t* restrict part_n = params->part_n; \
	dim_t m_max = params->m_max, n_max = params->n_max; \
	dim_t off_m0 = bli_auxinfo_off_m( auxinfo );\
	dim_t off_n0 = bli_auxinfo_off_n( auxinfo );\
	obj_t* C_local = params->local;\
	inc_t totaloff = ((char*)c - ((char*)(C_local->buffer)))/(sizeof(ctype)*2);\
	dim_t m0 = totaloff/rs_c;\
	dim_t n0 = totaloff%rs_c;\
\
	/*auxinfo_t auxinfo_r = *auxinfo; */\
    /*bli_auxinfo_set_params( params_r, &auxinfo_r ); */ \
\
	if ( !PASTEMAC(chr,eq0)( *alpha_i ) || \
	     !PASTEMAC(chr,eq0)( *beta_i ) || \
	     !bli_is_preferentially_stored( rs_c, cs_c, row_pref ) ) \
	{ \
		printf("Unsupported 1m case.\n");\
		bli_abort();\
	} \
	else \
	{ \
		/* In the typical cases, we use the real part of beta and*/\
		/* accumulate directly into the output matrix c. */ \
\
		c_use    = ( ctype_r* )c; \
		rs_c_use = rs_c; \
		cs_c_use = cs_c; \
\
		/* Convert the strides from being in units of complex elements to*/\
		/* be in units of real elements. Note that we don't need to check for*/\
		/* general storage here because that case corresponds to the scenario*/\
		/* where we are using the ct buffer and its rs_ct/cs_ct strides. */ \
		if ( !row_pref ) cs_c_use *= 2; \
		else             rs_c_use *= 2; \
\
\
		/* The following gemm micro-kernel call implements the 1m method,*/\
		/* which induces a complex matrix multiplication by calling the*/\
		/* real matrix micro-kernel on micro-panels that have been packed*/\
		/* according to the 1e and 1r formats. */ \
\
		/* c = beta * c + alpha_r * a * b; */ \
		rgemm_ukr \
		( \
		  m_r, \
		  n_r, \
		  k_r, \
		  alpha_r, \
		  a_r, \
		  b_r, \
		  beta_r, \
		  ct, rs_c_use, cs_c_use, \
		  &auxinfo, \
		  cntx  \
		); \
\
		for ( dim_t s = 0; s < nsplit; s++ ) \
		{ \
			\
			ctype_r lambda; \
			PASTEMAC3(chr, s, chr, scal2s)( *alpha_r, coef[ s ], lambda ); \
	\
			ctype_r* restrict c_use = ( ctype_r* )c + off_m[ s ] * rs_c_use + off_n[ s ] * cs_c_use; \
			\
			inc_t total_off_m = m0 + off_m[s];\
			inc_t total_off_n = n0 + off_n[s];\
			\
			/*dim_t m_use = bli_min(part_m[s], bli_min( m, m_max - total_off_m ));*/ \
			/*dim_t n_use = bli_min(part_n[s], bli_min( n, n_max - total_off_n ));*/ \
			dim_t m_use  = bli_max(0, bli_min(m_r, part_m[s] - m0 )); \
			dim_t n_use = bli_max(0, bli_min(n_r, part_n[s] - n0 )); \
			PASTEMAC(ch,axpbys_mxn)( m_use, n_use, \
			                         &lambda, ct, rs_ct, cs_ct, \
			                         ( void* )beta, c_use, rs_c_use, cs_c_use ); \
		} \
	} \
}

INSERT_GENTFUNCRO( gemm1m_fmm, BLIS_CNAME_INFIX, BLIS_REF_SUFFIX )