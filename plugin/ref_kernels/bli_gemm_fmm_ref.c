/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2023, The University of Texas at Austin
   Copyright (C) 2023, Southern Methodist University

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:
    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    - Neither the name(s) of the copyright holder(s) nor the names of its
      contributors may be used to endorse or promote products derived
      from this software without specific prior written permission.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
   HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
   DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
   THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/

#include "blis.h"
#include STRINGIFY_INT(../PASTEMAC(plugin,BLIS_PNAME_INFIX).h)

#define DEBUG_gemm 0

#undef  GENTFUNC
#define GENTFUNC( ctype, ch, opname, arch, suf ) \
\
void PASTEMAC3(ch,opname,arch,suf) \
     ( \
             dim_t  m, \
             dim_t  n, \
             dim_t  k, \
       const void*  alpha, \
       const void*  a, \
       const void*  b, \
       const void*  beta, \
             void*  c, inc_t rs_c, inc_t cs_c, \
             auxinfo_t* data, \
       const cntx_t*    cntx  \
     ) \
{ \
	const num_t       dt       = PASTEMAC(ch,type); \
	const gemm_ukr_ft ukr      = bli_cntx_get_ukr_dt( dt, BLIS_GEMM_UKR, cntx ); \
	const bool        row_pref = bli_cntx_get_ukr_prefs_dt( dt, BLIS_GEMM_UKR_ROW_PREF, cntx ); \
	const dim_t       MR       = bli_cntx_get_blksz_def_dt( dt, BLIS_MR, cntx ); \
	const dim_t       NR       = bli_cntx_get_blksz_def_dt( dt, BLIS_NR, cntx ); \
\
	      ctype       ab[ BLIS_STACK_BUF_MAX_SIZE / sizeof(ctype) ]; \
	const ctype*      zero     = PASTEMAC(ch,0); \
	const inc_t       rs_ab    = row_pref ? NR : 1; \
	const inc_t       cs_ab    = row_pref ? 1 : MR; \
\
	fmm_params_t* params = ( fmm_params_t* )bli_auxinfo_params( data ); \
	dim_t nsplit = params->nsplit; \
	float* restrict coef = ( float* )params->coef; \
	inc_t* restrict off_m = params->off_m; \
	inc_t* restrict off_n = params->off_n; \
	dim_t m_max = params->m_max, n_max = params->n_max; \
	dim_t off_m0 = bli_auxinfo_off_m( data );\
	dim_t off_n0 = bli_auxinfo_off_n( data );\
\
	/* Compute the AB product and store in a temporary buffer. */ \
	/* TODO: optimize passes where only one sub-matrix is written. */ \
	/* TODO: also optimize prefetching for multiple sub-matrices. */ \
	if (DEBUG_gemm) printf("== PRE MULT mr %d   nr %d \t rs %d   cs %d:\n", MR, NR, rs_ab, cs_ab);\
	ukr \
	( \
		MR, \
		NR, \
		k, \
		alpha, \
		a, \
		b, \
		zero, \
		ab, rs_ab, cs_ab, \
		data, \
		cntx \
	); \
\
	ctype* abp = (ctype*) ab;\
	if (DEBUG_gemm) printf("\nnsplit %d \t %f %f %f %f\n\n", nsplit, coef[0], coef[1], coef[2], coef[3]);\
	if (DEBUG_gemm) {\
		printf("MATRIX AB %d %d %d \t %d %d:\n", m, n, k, m_max, n_max);\
		for (dim_t g = 0; g < n; g++) {\
			printf("\n");\
			for (dim_t i = 0; i < m; i++) {\
				printf("%5.3f ", abp[i*rs_ab + g*cs_ab]);\
			}\
		}\
		printf("\n\n\n\n-------------------------------------\n");\
	}\
\
	for ( dim_t s = 0; s < nsplit; s++ ) \
	{ \
		\
		ctype alpha_cast, lambda; \
		if (true) {\
		alpha_cast = *( ctype* )alpha; \
		PASTEMAC3(ch, s, ch, scal2s)( alpha_cast, coef[ s ], lambda ); \
\
		ctype* restrict c_use = ( ctype* )c + off_m[ s ] * rs_c + off_n[ s ] * cs_c; \
		/*if (s==3) printf("\t\t TIME TO ADD AB to C w/ coef %f \t off %d %d prev value is %5.3f -> %5.3f\n", lambda, off_m[s]*rs_c, off_n[s]*cs_c, c_use[0]);*/\
\
		\
		dim_t m_use = bli_min( m, m_max - ( off_m0 + off_m[ s ] ) ); \
		dim_t n_use = bli_min( n, n_max - ( off_n0 + off_n[ s ] ) ); \
		if (false) printf("\t\t%d - m_use %d n_use %d \t off0   %d %d \t max %d %d\n", s, m_use, n_use, off_m0, off_n0, m_max, n_max);\
\
		if (DEBUG_gemm) printf("\n\n\t\tINFO lambda %5.3f \t offset %d %d\t use %d %d \t \t %d %d\t %d %d\n\n", lambda, off_m[ s ], off_n[ s ], m_use, n_use, m, n, m_max, n_max);\
\
		PASTEMAC(ch,axpbys_mxn)( m_use, n_use, \
		                         &lambda, ab, rs_ab, cs_ab, \
		                         ( void* )beta, c_use, rs_c, cs_c ); \
		}\
	} \
}

INSERT_GENTFUNC_BASIC( gemm_fmm, BLIS_CNAME_INFIX, BLIS_REF_SUFFIX )

