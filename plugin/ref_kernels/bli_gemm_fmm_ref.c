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
	inc_t* restrict part_m = params->part_m; \
	inc_t* restrict part_n = params->part_n; \
	dim_t m_max = params->m_max, k_max = params->n_max; \
	\
	\
	\
	inc_t off_md = 0;\
	inc_t off_nd = 0;\
	dim_t part_md = 0;\
	dim_t part_nd = 0;\
	\
	dim_t row_tilde = params->m_tilde; \
	dim_t col_tilde = params->n_tilde; \
	int num_row_part_whole = m_max % row_tilde; \
    if (m_max % row_tilde == 0) num_row_part_whole = 0; \
    dim_t row_part_size = m_max / row_tilde; \
\
    int num_col_part_whole = k_max % col_tilde; \
    if (k_max % col_tilde == 0) num_col_part_whole = 0; \
    dim_t col_part_size = k_max / col_tilde; \
	\
	\
	\
	if (0) {dim_t m_max = params->m_max, n_max = params->n_max;} \
	obj_t* C_local = params->local;\
	inc_t totaloff = ((char*)c - ((char*)(C_local->buffer)))/sizeof(ctype);\
	dim_t m0 = totaloff/rs_c;\
	dim_t n0 = totaloff%rs_c;\
\
	/* Compute the AB product and store in a temporary buffer. */ \
	/* TODO: optimize passes where only one sub-matrix is written. */ \
	/* TODO: also optimize prefetching for multiple sub-matrices. */ \
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
\
	for (int i = 0; i < row_tilde; i++) {\
		for (int j = 0; j < col_tilde; j++) {\
		if(0) return;\
\
		dim_t s = j + i * col_tilde;\
\
		ctype alpha_cast, lambda; \
		alpha_cast = *( ctype* )alpha; \
		PASTEMAC3(ch, s, ch, scal2s)( alpha_cast, coef[ s ], lambda ); \
		if (PASTECH2(bli_,ch,eq0)(lambda)) continue;\
\
		\
		\
		int whole_i = bli_min(i, num_row_part_whole);\
        int partial_i = bli_min(row_tilde - num_row_part_whole, i - whole_i);\
\
        int whole_j = bli_min(j, num_col_part_whole);\
        int partial_j = bli_min(col_tilde - num_col_part_whole, j - whole_j);\
\
        if (i < num_row_part_whole)\
            part_md = row_part_size + 1;\
        else\
            part_md= row_part_size;\
\
        if (j < num_col_part_whole)\
            part_nd = col_part_size + 1;\
        else\
            part_nd = col_part_size;\
\
        off_md = whole_i * (row_part_size + 1) + partial_i * row_part_size;\
\
        off_nd = whole_j * (col_part_size + 1) + partial_j * col_part_size;\
\
\
\
\
		ctype* restrict c_use = ( ctype* )c + off_md * rs_c + off_nd * cs_c; \
		\
		inc_t total_off_m = m0 + off_m[s];\
		inc_t total_off_n = n0 + off_n[s];\
		\
		dim_t m_use  = bli_max(0, bli_min(m, part_md - m0 )); \
		dim_t n_use = bli_max(0, bli_min(n, part_nd - n0 )); \
		PASTEMAC(ch,axpbys_mxn)( m_use, n_use, \
		                         &lambda, ab, rs_ab, cs_ab, \
		                         ( void* )beta, c_use, rs_c, cs_c ); \
		}\
	} \
}

INSERT_GENTFUNC_BASIC( gemm_fmm, BLIS_CNAME_INFIX, BLIS_REF_SUFFIX )


#define FILL_ZERO(ctype, ch)\
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
	dim_t m_max = params->m_max, k_max = params->n_max; \
	obj_t* C_local = params->local;\
	\
	\
	inc_t off_md = 0;\
	inc_t off_nd = 0;\
	dim_t part_md = 0;\
	dim_t part_nd = 0;\
	\
	dim_t row_tilde = params->m_tilde; \
	dim_t col_tilde = params->n_tilde; \
	int num_row_part_whole = m_max % row_tilde; \
    if (m_max % row_tilde == 0) num_row_part_whole = 0; \
    dim_t row_part_size = m_max / row_tilde; \
\
    int num_col_part_whole = k_max % col_tilde; \
    if (k_max % col_tilde == 0) num_col_part_whole = 0; \
    dim_t col_part_size = k_max / col_tilde; \
    \
	inc_t totaloff = ((char*)c - ((char*)(C_local->buffer)))/sizeof(ctype);\
	dim_t m0 = totaloff/rs_c;\
	dim_t n0 = totaloff%rs_c;\
\
	/* Compute the AB product and store in a temporary buffer. */ \
	/* TODO: optimize passes where only one sub-matrix is written. */ \
	/* TODO: also optimize prefetching for multiple sub-matrices. */ \
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


#define ADD(ctype, ch, split, coef)\
	{\
		int j = split % col_tilde;\
		int i = (split - j) / col_tilde;\
		\
		ctype alpha_cast, lambda; \
		alpha_cast = *( ctype* )alpha; \
		PASTEMAC3(ch, s, ch, scal2s)( alpha_cast, coef, lambda ); \
		if (!PASTECH2(bli_,ch,eq0)(lambda))\
		{\
			\
			\
			int whole_i = bli_min(i, num_row_part_whole);\
	        int partial_i = bli_min(row_tilde - num_row_part_whole, i - whole_i);\
	\
	        int whole_j = bli_min(j, num_col_part_whole);\
	        int partial_j = bli_min(col_tilde - num_col_part_whole, j - whole_j);\
	\
	        if (i < num_row_part_whole)\
	            part_md = row_part_size + 1;\
	        else\
	            part_md = row_part_size;\
	\
	        if (j < num_col_part_whole)\
	            part_nd = col_part_size + 1;\
	        else\
	            part_nd = col_part_size;\
	\
	        off_md = whole_i * (row_part_size + 1) + partial_i * row_part_size;\
	\
        	off_nd = whole_j * (col_part_size + 1) + partial_j * col_part_size;\
        \
        \
\
			ctype* restrict c_use = ( ctype* )c + off_md * rs_c + off_nd * cs_c; \
			\
			inc_t total_off_m = m0 + off_md;\
			inc_t total_off_n = n0 + off_nd;\
			\
			dim_t m_use  = bli_max(0, bli_min(m, part_md - m0 )); \
			dim_t n_use = bli_max(0, bli_min(n, part_nd - n0 )); \
			PASTEMAC(ch,axpbys_mxn)( m_use, n_use, \
			                         &lambda, ab, rs_ab, cs_ab, \
			                         ( void* )beta, c_use, rs_c, cs_c ); \
		}\
	}\


#undef  GENTFUNC
#define GENTFUNC( ctype, ch, opname, s0, coef0 ) \
\
void PASTECH(ch,opname) \
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
{\
	FILL_ZERO(ctype, ch) \
	ADD(ctype, ch, s0, coef0) \
}


INSERT_GENTFUNC_BASIC(CLASSICAL_UKR_0, 0, 1)
INSERT_GENTFUNC_BASIC(CLASSICAL_UKR_1, 0, 1)
INSERT_GENTFUNC_BASIC(CLASSICAL_UKR_2, 1, 1)
INSERT_GENTFUNC_BASIC(CLASSICAL_UKR_3, 1, 1)
INSERT_GENTFUNC_BASIC(CLASSICAL_UKR_4, 2, 1)
INSERT_GENTFUNC_BASIC(CLASSICAL_UKR_5, 2, 1)
INSERT_GENTFUNC_BASIC(CLASSICAL_UKR_6, 3, 1)
INSERT_GENTFUNC_BASIC(CLASSICAL_UKR_7, 3, 1)

INSERT_GENTFUNC_BASIC(FMM_222_UKR_5, 3, 1)
INSERT_GENTFUNC_BASIC(FMM_222_UKR_6, 0, 1)


#undef  GENTFUNC
#define GENTFUNC( ctype, ch, opname, s0, coef0, s1, coef1 ) \
\
void PASTECH(ch,opname) \
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
{\
	FILL_ZERO(ctype, ch) \
	ADD(ctype, ch, s0, coef0) \
	{\
	ADD(ctype, ch, s1, coef1) \
	}\
}

INSERT_GENTFUNC_BASIC(FMM_222_UKR_0, 0, 1, 3, 1)
INSERT_GENTFUNC_BASIC(FMM_222_UKR_1, 2, 1, 3, -1)
INSERT_GENTFUNC_BASIC(FMM_222_UKR_2, 1, 1, 3, 1)
INSERT_GENTFUNC_BASIC(FMM_222_UKR_3, 0, 1, 2, 1)
INSERT_GENTFUNC_BASIC(FMM_222_UKR_4, 0, -1, 1, 1)
