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

// ctype* cc = (ctype*) c_use;\
// 	printf("\n\nMATRIX CCC: %5.3f\n", lambda2);\
// 	for (dim_t g = 0; g < panel_dim_use; g++) {\
// 		printf("\n");\
// 		for (dim_t i = 0; i < panel_len_use; i++) {\
// 			printf("%5.3f ", c_use[i*ldc + g*incc]);\
// 		}\
// 	}\
// 	printf("\n\n");\

#include "blis.h"
#include <complex.h>
#include STRINGIFY_INT(../PASTEMAC(plugin,BLIS_PNAME_INFIX).h)

#undef  GENTFUNC
#define GENTFUNC( ctype, ch, opname, arch, suf ) \
\
void PASTEMAC3(ch,opname,arch,suf) \
     ( \
             struc_t strucc, \
             diag_t  diagc, \
             uplo_t  uploc, \
             conj_t  conjc, \
             pack_t  schema, \
             bool    invdiag, \
             dim_t   panel_dim, \
             dim_t   panel_len, \
             dim_t   panel_dim_max, \
             dim_t   panel_len_max, \
             dim_t   panel_dim_off, \
             dim_t   panel_len_off, \
             dim_t   panel_bcast, \
       const void*   kappa, \
       const void*   c, inc_t incc, inc_t ldc, \
             void*   p,             inc_t ldp, \
       const void*   params_, \
       const cntx_t* cntx  \
     ) \
{ \
	const num_t dt = PASTEMAC(ch,type); \
\
	fmm_params_t*    params    = ( fmm_params_t* )params_; \
	packm_cxk_ker_ft packm_def = bli_cntx_get_ukr_dt( dt, BLIS_PACKM_KER, cntx ); \
\
	dim_t nsplit = params->nsplit; \
	float* restrict coef = ( float* )params->coef; \
	float* coeff = (float*) params->coef;\
	inc_t* restrict off_m = params->off_m; \
	inc_t* restrict off_k = params->off_n; \
	dim_t* restrict part_m = params->part_m;\
	dim_t* restrict part_n = params->part_n; \
	dim_t m_max = params->m_max, k_max = params->n_max; \
\
	/* The first sub-matrix also needs a coefficient and offset computation. */ \
	ctype kappa_cast, lambda; \
	kappa_cast = *( ctype* )kappa; \
	PASTEMAC3(ch, s, ch,scal2s)( kappa_cast, coef[ 0 ], lambda ); \
	ctype lambda2 = lambda;\
\
	const ctype* restrict c_use = ( ctype* )c + off_m[ 0 ] * incc + off_k[ 0 ] * ldc; \
	      ctype* restrict p_use = ( ctype* )p; \
\
	/* Check if we need to shrink the micro-panel due to unequal partitioning. */ \
	dim_t panel_dim_use = bli_min( panel_dim, m_max - ( panel_dim_off + off_m[ 0 ] ) ); \
	dim_t panel_len_use = bli_min( panel_len, k_max - ( panel_len_off + off_k[ 0 ] ) ); \
\
	\
	/* Call the usual packing kernel to pack the first sub-matrix and take
	   care zeroing out the edges. */ \
\
	packm_def \
	( \
	  conjc, \
	  schema, \
	  panel_dim, \
	  panel_dim_max, \
	  1, \
	  panel_len, \
	  panel_len_max, \
	  &lambda, \
	  c_use, incc, ldc, \
	  p_use,       ldp, \
	  params, \
	  cntx \
	); \
\
	for ( dim_t s = 1; s < nsplit; s++ ) \
	{ \
		PASTEMAC3(ch, s, ch, scal2s)( kappa_cast, coef[ s ], lambda );\
\
		inc_t total_off_m = panel_dim_off + off_m[s];\
		inc_t total_off_n = panel_len_off + off_k[s];\
\
		c_use = ( ctype* )c + off_m[s] * incc + off_k[s] * ldc; \
		p_use = ( ctype* )p; \
\
		/* Check if we need to shrink the micro-panel due to unequal partitioning. */ \
		panel_dim_use = bli_min(panel_dim, part_m[s] - panel_dim_off ); \
		panel_len_use = bli_min(panel_len, part_n[s] - panel_len_off ); \
		\
		/* For subsequence sub-matrices, we don't need to re-zero any edges, just accumulate. */ \
		for ( dim_t j = 0; j < panel_len_use; j++ ) \
		{ \
			for ( dim_t i = 0; i < panel_dim_use; i++ ) \
			for ( dim_t d = 0; d < panel_bcast; d++ ) \
			{ \
				PASTEMAC(ch,axpys)( lambda, c_use[ i*incc ], p_use[ i*panel_bcast + d ] ); \
			} \
			c_use += ldc; \
			p_use += ldp; \
		} \
	} \
}

INSERT_GENTFUNC_BASIC( packm_fmm, BLIS_CNAME_INFIX, BLIS_REF_SUFFIX )
