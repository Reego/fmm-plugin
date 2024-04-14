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

//
// Parameters passed to the plugin registration and initialization
// functions.
//

// Use global kernel id variables instead of passing by argument

extern siz_t FMM_BLIS_PACK_UKR;
extern siz_t FMM_BLIS_PACK_UKR_SYMM;
extern siz_t FMM_BLIS_GEMM_UKR;
extern siz_t FMM_BLIS_GEMM1M_UKR;

#define plugin_fmm_blis_params
#define plugin_fmm_blis_params_only

//
// Parameter structures for kernels
//

#define FMM_BLIS_MULTS 7
#define MAX_NUM_PARTS 16

typedef struct fmm_s {

    int m_tilde;
    int n_tilde;
    int k_tilde;
    int R;

    int* U;
    int* V;
    int* W;
} fmm_t;

// The same structure is used for packing and in the micro-kernel, but
// each packing node and the micro-kernel each get a separate instance with distinct
// sub-matrices and coefficients.
typedef struct fmm_params_t
{
    // int id;
	// number of partitions to pack or accumulate
	dim_t nsplit;

	// coefficient for each partition (in the computational datatype)
	float coef[ MAX_NUM_PARTS ];

    // size for each partition (in the computational datatype)
    dim_t part_m[ MAX_NUM_PARTS ];
    dim_t part_n[ MAX_NUM_PARTS ];

	// offsets of each partition relative to the parent matrix
	// (when packing, m is the "short micro-panel dimension (m or n)", and n
	// is the "long micro-panel dimension (k)")
	inc_t off_m[ MAX_NUM_PARTS ], off_n[ MAX_NUM_PARTS ];

	// also keep track of the total matrix size so that we can detect sub-matrix
	// edge cases
	dim_t m_max, n_max;

    //
    obj_t* local;
} fmm_params_t;

struct fmm_cntl_s
{
    cntl_t cntl; // cntl field must be present and come first.
    fmm_t* fmm;
};
typedef struct fmm_cntl_s fmm_cntl_t;

struct fmm_gemm_cntl_s
{
    fmm_cntl_t fmm_cntl;
    gemm_cntl_t gemm_cntl;
};
typedef struct fmm_gemm_cntl_s fmm_gemm_cntl_t;

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

//
// Prototypes for reference kernels
//

#undef GENTPROT
#define GENTPROT( ctype, ch, config_infix ) \
\
void PASTEMAC3(ch,packm_fmm,config_infix,BLIS_REF_SUFFIX) \
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
       const void*   params, \
       const cntx_t* cntx  \
     ); \
\
void PASTEMAC3(ch,packm_symm_fmm,config_infix,BLIS_REF_SUFFIX) \
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
       const void*   params, \
       const cntx_t* cntx  \
     ); \
\
void PASTEMAC3(ch,gemm_fmm,config_infix,BLIS_REF_SUFFIX) \
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
     );

#undef GENTFUNCRO
#define GENTFUNCRO( ctype_r, chr, config_infix ) \
\
void PASTEMAC3(chr,gemm1m_fmm,config_infix,BLIS_REF_SUFFIX) \
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
     ); \

// Generate reference kernel prototypes for each configuration AND data type
#undef GENTCONF
#define GENTCONF( CONFIG, config ) \
\
INSERT_GENTPROT_BASIC( PASTECH(_,config) )\
INSERT_GENTFUNCRO_BASIC( PASTECH(_,config) )


INSERT_GENTCONF

//
// Registration and intialization function prototypes.
//

#undef GENTCONF
#define GENTCONF( CONFIG, config ) \
\
void PASTEMAC3(plugin_init,BLIS_PNAME_INFIX,_,config)( PASTECH2(plugin,BLIS_PNAME_INFIX,_params) ); \
void PASTEMAC4(plugin_init,BLIS_PNAME_INFIX,_,config,BLIS_REF_SUFFIX)( PASTECH2(plugin,BLIS_PNAME_INFIX,_params) );

INSERT_GENTCONF

BLIS_EXPORT_BLIS err_t PASTEMAC(plugin_register,BLIS_PNAME_INFIX)( PASTECH2(plugin,BLIS_PNAME_INFIX,_params) );

