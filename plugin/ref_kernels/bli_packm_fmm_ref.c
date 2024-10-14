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


//         int whole_i = bli_min(i, num_row_part_whole);\
//         int partial_i = bli_min(row_tilde - num_row_part_whole, i - whole_i);\
// \
//         int whole_j = bli_min(j, num_col_part_whole);\
//         int partial_j = bli_min(col_tilde - num_col_part_whole, j - whole_j);\
// \
//         if (i < num_row_part_whole)\
//             part_md = row_part_size + 1;\
//         else\
//             part_md= row_part_size;\
// \
//         if (j < num_col_part_whole)\
//             part_nd = col_part_size + 1;\
//         else\
//             part_nd = col_part_size;\
// \
//         off_md = whole_i * (row_part_size + 1) + partial_i * row_part_size;\
// \
//         off_kd = whole_j * (col_part_size + 1) + partial_j * col_part_size;\
//         \
//         if (ldp == 8)\
//         {\
//         	inc_t temp0;\
//         	temp0 = off_kd;\
//         	off_kd = off_md;\
//         	off_md = temp0;\
//         	\
//         	dim_t temp1 = part_md;\
//         	part_md = part_nd;\
//         	part_nd = temp1;\
//         }\


#include "blis.h"
#include <complex.h>
#include <time.h>
#include STRINGIFY_INT(../PASTEMAC(plugin,BLIS_PNAME_INFIX).h)

#define BLIS_ASM_SYNTAX_ATT
#include "bli_x86_asm_macros.h"

void bli_daxpys_mxn(
	const dim_t m0,
	const dim_t n,
	double* restrict alpha,
    double* restrict x,
    const inc_t rs_x0,
    const inc_t cs_x0,
    double* restrict beta,
    double* restrict y,
    const inc_t rs_y0,
    const inc_t cs_y0
    )
{
	// This is the panel dimension assumed by the packm kernel.
	const dim_t      mr    = 6;
	const dim_t      nr    = 8;

	// Typecast local copies of integers in case dim_t and inc_t are a
	// different size than is expected by load instructions.

	const uint64_t m = m0;

	const uint64_t n_full = n / 4;
	const uint64_t n_partial = (n % 4) / 2;
	const uint64_t n_left = (n % 4) % 2;

	const uint64_t rs_x = rs_x0;
	const uint64_t cs_x = cs_x0;

	const uint64_t rs_y = rs_y0;
	const uint64_t cs_y = cs_y0;

	const double one = 1.0;

	// -------------------------------------------------------------------------

	//
	// n is leading dimension
	// row major for some reason
	//
	// printf("huh %d %d %d %d\n", m, n, n_full, n_partial);
	if (m == 6 && n == 8 && cs_x == 1 && cs_y == 1)
	{
		// printf("m %d \t n %d \t %d %d %d %5.2f \t %ld %ld %ld %ld\n", m, n, n_full, n_partial, n_left, *((double*) alpha), rs_x, rs_y, cs_x, cs_y);
		begin_asm()

		mov(var(n_full), r13)
		mov(var(n_partial), r14)

		mov(var(rs_x), r8)                 // load rs_x
		mov(var(cs_x), r10)                // load cs_x
		lea(mem(, r8, 8), r8)              // rs_x *= sizeof(double)
		lea(mem(, r10, 8), r10)            // cs_x *= sizeof(double)

		mov(var(rs_y), r11)                // load rs_y
		mov(var(cs_y), r15)                // load cs_y
		lea(mem(, r11, 8), r11)			   // rs_y *= sizeof(double)
		lea(mem(, r15, 8), r15)            // cs_y *= sizeof(double)

		// lea(mem(   , r10, 4), r14)         // r14 = 4*lda

		mov(var(alpha), rcx)               // load address of alpha
		vbroadcastsd(mem(rcx), ymm8)       // broadcast alpha

		// -- kappa unit, column storage on A --------------------------------------

		label(.DCOLUNIT)

		mov(var(x), rax)
		mov(var(y), rbx)

		mov(var(m), rsi)

		label(.DMFULLLOOP)

		mov(rax, rcx)
		mov(rbx, rdx)

		vmovupd(mem(rax,         0), ymm1)						   // load C block
		vmovupd(mem(rbx,         0), ymm2)						   // load buffer
		vfmadd132pd(ymm8, ymm2, ymm1)

		lea(mem(rax, r10, 4), rax)
		add(imm(4*8), rbx)

		vmovupd(mem(rax, 0), ymm3)						   // load C block
		vmovupd(mem(rbx, 0), ymm4)						   // load buffer
		vfmadd132pd(ymm8, ymm4, ymm3)
		vmovupd(ymm1, mem(rdx, 0))
		vmovupd(ymm3, mem(rbx, 0))
		
		add(r8, rcx)
		add(r11, rdx)
		mov(rcx, rax)
		mov(rdx, rbx)

		dec(rsi)
		jne(.DMFULLLOOP)                   // iterate again if i != 0.

		label(.DDONE)

		end_asm(
		: // output operands (none)
		: // input operands
		  [n_full] "m" (n_full),
		  [n_partial] "m" (n_partial),
		  [x]      "m" (x),
		  [rs_x]   "m" (rs_x),
		  [cs_x]   "m" (cs_x),
		  [y]      "m" (y),
		  [rs_y]   "m" (rs_y),
		  [cs_y]   "m" (cs_y),
		  [alpha]  "m"  (alpha),
		  [m]	   "m" (m)
		: // register clobber list
		  "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
		  "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
		  "xmm0", "xmm1", "xmm2", "xmm3",
		  "xmm4", "xmm5", "xmm6", "xmm7",
		  "xmm8", "xmm9", "xmm10", "xmm11",
		  "xmm12", "xmm13", "xmm14", "xmm15",
		  "memory"
		)
	}
	else {
		bli_daxpbys_mxn( m, n,
		                  alpha, x, rs_x0, cs_x0,
		                  &one, y, rs_y0, cs_y0 );
		return;
	}

	if (n_left > 0) {
		const uint64_t n_used = n_full * 4 + n_partial * 2;
		// const uint64_t n_used = 0;
		// const uint64_t _n_left = n;
		bli_daxpbys_mxn( m, n_left,
		                  alpha,
		                  x + cs_x0 * n_used, rs_x0, cs_x0,
		                  &one, y + cs_y0 * n_used, rs_y0, cs_y0
		                );
	}
}

#undef  GENTFUNC
#define GENTFUNC( ctype, ch, opname, arch, suf, func ) \
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
	dim_t m_max = params->m_max; \
	dim_t k_max = params->n_max; \
\
	/* The first sub-matrix also needs a coefficient and offset computation. */ \
	ctype kappa_cast, lambda; \
	kappa_cast = *( ctype* )kappa; \
	PASTEMAC3(ch, s, ch, scal2s)( kappa_cast, coef[ 0 ], lambda ); \
\
	inc_t off_md = 0; \
	inc_t off_kd = 0; \
	\
	dim_t part_md = 0; \
	dim_t part_nd = 0; \
	\
	\
	\
	\
	dim_t row_tilde = params->m_tilde; \
	dim_t col_tilde = params->n_tilde; \
	if (ldp == 8) { \
		dim_t temp = m_max;\
		m_max = k_max;\
		k_max = temp;\
	} \
	int num_row_part_whole = m_max % row_tilde; \
    if (m_max % row_tilde == 0) num_row_part_whole = 0; \
    dim_t row_part_size = m_max / row_tilde; \
    if(0)printf("row-part-size: %d and %d\n", row_part_size, num_row_part_whole);\
\
    int num_col_part_whole = k_max % col_tilde; \
    if (k_max % col_tilde == 0) num_col_part_whole = 0; \
    dim_t col_part_size = k_max / col_tilde; \
    if(0)printf("col-part-size: %d and %d\n", col_part_size, num_col_part_whole);\
	\
	\
	\
	\
\
	const ctype* restrict c_use = ( ctype* )c + off_md * incc + off_kd * ldc; \
	      ctype* restrict p_use = ( ctype* )p; \
\
	/* Check if we need to shrink the micro-panel due to unequal partitioning. */ \
	dim_t panel_dim_use = bli_min( panel_dim, m_max - ( panel_dim_off + off_md ) ); \
	dim_t panel_len_use = bli_min( panel_len, k_max - ( panel_len_off + off_kd ) ); \
\
	if(0)printf("ldp %d - panel_dim_use %d - panel_len_use %d\n", ldp, panel_dim_use, panel_len_use);\
	\
	/* Call the usual packing kernel to pack the first sub-matrix and take care zeroing out the edges. */ \
\
	if (0 && PASTECH2(bli_,ch,eq0)(lambda) && 0) {\
		panel_dim = 0;\
		panel_len = 0;\
		printf("IT'S ZERO\n\n");\
	}\
	\
	packm_def\
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
	for (int i = 0; i < row_tilde; i++) {\
        for (int j = 0; j < col_tilde; j++) {\
        \
        dim_t s = j + i * col_tilde;\
        if (s == 0) continue;\
        PASTEMAC3(ch, s, ch, scal2s)( kappa_cast, coef[ s ], lambda );\
\
		if (PASTECH2(bli_,ch,eq0)(lambda)) continue;\
        \
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
            part_md = row_part_size;\
\
        if (j < num_col_part_whole)\
            part_nd = col_part_size + 1;\
        else\
            part_nd = col_part_size;\
\
        off_md = whole_i * (row_part_size + 1) + partial_i * row_part_size;\
\
        off_kd = whole_j * (col_part_size + 1) + partial_j * col_part_size;\
        \
        if (ldp == 8)\
        {\
        	inc_t temp0;\
        	temp0 = off_kd;\
        	off_kd = off_md;\
        	off_md = temp0;\
        	\
        	dim_t temp1 = part_md;\
        	part_md = part_nd;\
        	part_nd = temp1;\
        }\
    \
        \
        \
\
		\
		\
		\
		\
\
		inc_t total_off_m = panel_dim_off + off_md;\
		inc_t total_off_n = panel_len_off + off_kd;\
\
				/* Check if we need to shrink the micro-panel due to unequal partitioning. */ \
		panel_dim_use = bli_min(panel_dim, part_md - panel_dim_off ); \
		panel_len_use = bli_min(panel_len, part_nd - panel_len_off ); \
		if(0) printf("| ldp %d - panel_dim_use %d - panel_len_use %d \t %d\n", ldp, panel_dim_use, panel_len_use, s);\
\
		if (!params->reindex) {\
			c_use = ( ctype* )c + off_md * incc + off_kd * ldc; \
			p_use = ( ctype* )p; \
		}\
		else {\
			ldc = bli_obj_col_stride( &(params->parts[s]) );\
			incc = bli_obj_row_stride( &(params->parts[s]) );\
			c_use = ( ctype* )params->parts[s].buffer + panel_dim_off * incc + panel_len_off * ldc; \
			p_use = ( ctype* )p; \
		}\
		\
		/* For subsequence sub-matrices, we don't need to re-zero any edges, just accumulate. */ \
		if(1){\
\
			if(0) {\
				/* ldp and then 1? */ \
				if(0) printf("ldp is %ld and %ld and %ld %ld\n\n", ldp, panel_dim_use, ldc, incc);\
				if (1) func(panel_dim_use, panel_len_use, &lambda, c_use, incc, ldc, &BLIS_ONE, p_use, 1, ldp);\
				continue;\
			}\
\
			else {\
				for ( dim_t j = 0; j < panel_len_use; j++ ) \
				{ \
					for ( dim_t i = 0; i < panel_dim_use; i++ ) {\
						PASTEMAC(ch,axpys)( lambda, c_use[ i*incc ], p_use[ i ] ); \
					}\
					c_use += ldc; \
					p_use += ldp; \
				} \
			}\
		}\
		else {\
		}\
	}} \
}

			// func( panel_dim_use, panel_len_use, \
		 //                         &lambda, c_use, incc, ldc, \
		 //                         &BLIS_ONE, p_use, 1, ldp ); \

GENTFUNC( float,    s, packm_fmm, BLIS_CNAME_INFIX, BLIS_REF_SUFFIX, bli_saxpbys_mxn) \
GENTFUNC( double,   d, packm_fmm, BLIS_CNAME_INFIX, BLIS_REF_SUFFIX, bli_daxpys_mxn) \
GENTFUNC( scomplex, c, packm_fmm, BLIS_CNAME_INFIX, BLIS_REF_SUFFIX, bli_caxpbys_mxn) \
GENTFUNC( dcomplex, z, packm_fmm, BLIS_CNAME_INFIX, BLIS_REF_SUFFIX, bli_zaxpbys_mxn)

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
\
	const ctype* restrict c_use = ( ctype* )c + off_m[ 0 ] * incc + off_k[ 0 ] * ldc; \
	      ctype* restrict p_use = ( ctype* )p; \
\
	/* Check if we need to shrink the micro-panel due to unequal partitioning. */ \
	dim_t panel_dim_use = bli_min( panel_dim, m_max - ( panel_dim_off + off_m[ 0 ] ) ); \
	dim_t panel_len_use = bli_min( panel_len, k_max - ( panel_len_off + off_k[ 0 ] ) ); \
	inc_t ldc_prime = bli_obj_col_stride( &(params->parts[0]) );\
	inc_t incc_prime = bli_obj_row_stride( &(params->parts[0]) );\
	inc_t ldc_prev = ldc;\
	inc_t incc_prev = incc;\
	if (params->reindex) {\
		incc = incc_prime;\
		ldc = ldc_prime;\
		c_use = ( ctype* )params->parts[0].buffer + panel_dim_off * incc + panel_len_off * ldc;\
	}\
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
		if (PASTECH2(bli_,ch,eq0)(lambda)) continue;\
\
		inc_t total_off_m = panel_dim_off + off_m[s];\
		inc_t total_off_n = panel_len_off + off_k[s];\
\
				/* Check if we need to shrink the micro-panel due to unequal partitioning. */ \
		panel_dim_use = bli_min(panel_dim, part_m[s] - panel_dim_off ); \
		panel_len_use = bli_min(panel_len, part_n[s] - panel_len_off ); \
		\
		/*printf("%d panel_dim_use, panel_len_use %d %d\n", ldp, panel_dim_use, panel_len_use);*/\
\
		if (!params->reindex) {\
			c_use = ( ctype* )c + off_m[s] * incc + off_k[s] * ldc; \
			p_use = ( ctype* )p; \
		}\
		else {\
			ldc = bli_obj_col_stride( &(params->parts[s]) );\
			incc = bli_obj_row_stride( &(params->parts[s]) );\
			c_use = ( ctype* )params->parts[s].buffer + panel_dim_off * incc + panel_len_off * ldc; \
			p_use = ( ctype* )p; \
		}\
		\
		if (0) printf("| ldp %d - panel_dim_use %d - panel_len_use %d \t %d\n", ldp, panel_dim_use, panel_len_use, s);\
		if (0) printf("%d panel_dim_use, panel_len_use %d %d\n", ldp, panel_dim_use, panel_len_use);\
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
	\
}

INSERT_GENTFUNC_BASIC( packm_fmm_m, BLIS_CNAME_INFIX, BLIS_REF_SUFFIX )



#define ADD( ctype, ch, split, coef ) \
	{\
		int j = split % col_tilde;\
		int i = (split - j) / col_tilde;\
		PASTEMAC3(ch, s, ch, scal2s)( kappa_cast, coef, lambda );\
\
		if (!PASTECH2(bli_,ch,eq0)(lambda))\
		{\
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
\
        off_md = whole_i * (row_part_size + 1) + partial_i * row_part_size;\
\
        off_kd = whole_j * (col_part_size + 1) + partial_j * col_part_size;\
        \
        if (ldp == 8)\
        {\
        	inc_t temp0;\
        	temp0 = off_kd;\
        	off_kd = off_md;\
        	off_md = temp0;\
        	\
        	dim_t temp1 = part_md;\
        	part_md = part_nd;\
        	part_nd = temp1;\
        }\
    \
        \
        \
\
			inc_t total_off_m = panel_dim_off + off_md;\
			inc_t total_off_n = panel_len_off + off_kd;\
	\
					/* Check if we need to shrink the micro-panel due to unequal partitioning. */ \
			panel_dim_use = bli_min(panel_dim, part_md - panel_dim_off ); \
			panel_len_use = bli_min(panel_len, part_nd - panel_len_off ); \
	\
			if (!params->reindex) {\
				c_use = ( ctype* )c + off_md * incc + off_kd * ldc; \
				p_use = ( ctype* )p; \
			}\
			else {\
				ldc = bli_obj_col_stride( &(params->parts[split]) );\
				incc = bli_obj_row_stride( &(params->parts[split]) );\
				c_use = ( ctype* )params->parts[split].buffer + panel_dim_off * incc + panel_len_off * ldc; \
				p_use = ( ctype* )p; \
			}\
			\
			/* For subsequence sub-matrices, we don't need to re-zero any edges, just accumulate. */ \
			if(1)\
			{for ( dim_t j = 0; j < panel_len_use; j++ ) \
			{ \
				for ( dim_t i = 0; i < panel_dim_use; i++ ) \
				for ( dim_t d = 0; d < panel_bcast; d++ ) \
				{ \
					if(0)printf("%5.2g \n", c_use[i*incc]);\
					PASTEMAC(ch,axpys)( lambda, c_use[ i*incc ], p_use[ i*panel_bcast + d ] ); \
				} \
				c_use += ldc; \
				p_use += ldp; \
			}\
			}\
			else {\
				PASTEMAC(ch,axpbys_mxn)( panel_dim_use, panel_len_use, \
		                         &lambda, c_use, incc, ldc, \
		                         &BLIS_ONE, p_use, 1, ldp ); \
			}\
		}\
	}\


#define FILL_ZERO(ctype, ch, split, coef) \
	const num_t dt = PASTEMAC(ch,type); \
\
	fmm_params_t*    params    = ( fmm_params_t* )params_; \
	packm_cxk_ker_ft packm_def = bli_cntx_get_ukr_dt( dt, BLIS_PACKM_KER, cntx ); \
\
	dim_t m_max = params->m_max, k_max = params->n_max; \
\
	/* The first sub-matrix also needs a coefficient and offset computation. */ \
	ctype kappa_cast, lambda; \
	kappa_cast = *( ctype* )kappa; \
	PASTEMAC3(ch, s, ch, scal2s)( kappa_cast, coef, lambda ); \
	\
	inc_t off_md = 0; \
	inc_t off_kd = 0; \
	\
	dim_t part_md = 0; \
	dim_t part_nd = 0; \
	\
	\
	\
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
	int j = split % col_tilde;\
	int i = (split - j) / col_tilde;\
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
\
    off_md = whole_i * (row_part_size + 1) + partial_i * row_part_size;\
\
    off_kd = whole_j * (col_part_size + 1) + partial_j * col_part_size;\
    \
    if (ldp == 8)\
    {\
    	inc_t temp0;\
    	temp0 = off_kd;\
    	off_kd = off_md;\
    	off_md = temp0;\
    	\
    	dim_t temp1 = part_md;\
    	part_md = part_nd;\
    	part_nd = temp1;\
    }\
\
    \
    \
\
	inc_t total_off_m = panel_dim_off + off_md;\
	inc_t total_off_n = panel_len_off + off_kd;\
\
			/* Check if we need to shrink the micro-panel due to unequal partitioning. */ \
	dim_t panel_dim_use = bli_min(panel_dim, part_md - panel_dim_off ); \
	dim_t panel_len_use = bli_min(panel_len, part_nd - panel_len_off ); \
	\
\
	const ctype* restrict c_use = ( ctype* )c + off_md * incc + off_kd * ldc; \
	      ctype* restrict p_use = ( ctype* )p; \
\
	/* Check if we need to shrink the micro-panel due to unequal partitioning. */ \
	if (0) printf("%d %d %5.2g\n", panel_dim_use, panel_len_use, lambda);\
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

#undef  GENTFUNC
#define GENTFUNC( ctype, ch, opname, s0, coef0, s1, coef1 ) \
\
void PASTECH(ch,opname) \
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
{\
	FILL_ZERO(ctype, ch, s0, coef0) \
	ADD(ctype, ch, s1, coef1) \
}

INSERT_GENTFUNC_BASIC(CLASSICAL_PACK_B_0, 0, 1, 0, 0)
INSERT_GENTFUNC_BASIC(CLASSICAL_PACK_B_1, 2, 1, 0, 0)
INSERT_GENTFUNC_BASIC(CLASSICAL_PACK_B_2, 1, 1, 0, 0)
INSERT_GENTFUNC_BASIC(CLASSICAL_PACK_B_3, 3, 1, 0, 0)
INSERT_GENTFUNC_BASIC(CLASSICAL_PACK_B_4, 0, 1, 0, 0)
INSERT_GENTFUNC_BASIC(CLASSICAL_PACK_B_5, 2, 1, 0, 0)
INSERT_GENTFUNC_BASIC(CLASSICAL_PACK_B_6, 1, 1, 0, 0)
INSERT_GENTFUNC_BASIC(CLASSICAL_PACK_B_7, 3, 1, 0, 0)

INSERT_GENTFUNC_BASIC(CLASSICAL_PACK_A_0, 0, 1, 0, 0)
INSERT_GENTFUNC_BASIC(CLASSICAL_PACK_A_1, 1, 1, 0, 0)
INSERT_GENTFUNC_BASIC(CLASSICAL_PACK_A_2, 0, 1, 0, 0)
INSERT_GENTFUNC_BASIC(CLASSICAL_PACK_A_3, 1, 1, 0, 0)
INSERT_GENTFUNC_BASIC(CLASSICAL_PACK_A_4, 2, 1, 0, 0)
INSERT_GENTFUNC_BASIC(CLASSICAL_PACK_A_5, 3, 1, 0, 0)
INSERT_GENTFUNC_BASIC(CLASSICAL_PACK_A_6, 2, 1, 0, 0)
INSERT_GENTFUNC_BASIC(CLASSICAL_PACK_A_7, 3, 1, 0, 0)

//

INSERT_GENTFUNC_BASIC(FMM_222_PACK_A_0, 0, 1, 3, 1)
INSERT_GENTFUNC_BASIC(FMM_222_PACK_A_1, 2, 1, 3, 1)
INSERT_GENTFUNC_BASIC(FMM_222_PACK_A_2, 0, 1, 0, 0)
INSERT_GENTFUNC_BASIC(FMM_222_PACK_A_3, 3, 1, 0, 0)
INSERT_GENTFUNC_BASIC(FMM_222_PACK_A_4, 0, 1, 1, 1)
INSERT_GENTFUNC_BASIC(FMM_222_PACK_A_5, 0, -1, 2, 1)
INSERT_GENTFUNC_BASIC(FMM_222_PACK_A_6, 1, 1, 3, -1)

INSERT_GENTFUNC_BASIC(FMM_222_PACK_B_0, 0, 1, 3, 1)
INSERT_GENTFUNC_BASIC(FMM_222_PACK_B_1, 0, 1, 0, 0)
INSERT_GENTFUNC_BASIC(FMM_222_PACK_B_2, 1, 1, 3, -1)
INSERT_GENTFUNC_BASIC(FMM_222_PACK_B_3, 0, -1, 2, 1)
INSERT_GENTFUNC_BASIC(FMM_222_PACK_B_4, 3, 1, 0, 0)
INSERT_GENTFUNC_BASIC(FMM_222_PACK_B_5, 0, 1, 1, 1)
INSERT_GENTFUNC_BASIC(FMM_222_PACK_B_6, 2, 1, 3, 1)
