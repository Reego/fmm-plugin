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

//  ctype* cc = (ctype*) c_use;\
// 	printf("\n\nMATRIX CCC: %5.3f\n", lambda2);\
// 	for (dim_t g = 0; g < panel_dim_use; g++) {\
// 		printf("\n");\
// 		for (dim_t i = 0; i < panel_len_use; i++) {\
// 			printf("%5.3f ", c_use[i*ldc + g*incc]);\
// 		}\
// 	}

#include "blis.h"
#include <complex.h>
#include STRINGIFY_INT(../PASTEMAC(plugin,BLIS_PNAME_INFIX).h)

#define PACKM_1E_BODY( ctype, ch, pragma, cdim, dfac, inca2, op ) \
\
do \
{ \
  for ( dim_t k = 0; k < panel_len; k++ ) \
  { \
    pragma \
    for ( dim_t i = 0; i < cdim; i++ ) \
    for ( dim_t d = 0; d < dfac; d++ ) \
    { \
      dim_t panel_dim_total_off = panel_dim + i;\
      dim_t panel_len_total_off = panel_len + k;\
      doff_t diagoffc = panel_dim_total_off - panel_len_total_off;\
      ctype* c_use = (c1 + i * incc2); \
      if ( bli_is_triangular( strucc )) {\
          if (diagoffc < 0 && bli_is_upper( uploc ) || \
              diagoffc > 0 && bli_is_lower( uploc )) {\
              c_use = c_begin + panel_len_total_off * incc2 + panel_dim_total_off * ldc2;\
          }\
      }\
      if (acc == 0) {\
        PASTEMAC(ch,scal2ris)(  kappa_r, kappa_i, *(c_use + 0), *(c_use + 1), \
                                          *(pi1_ri + (i*2 + 0)*dfac + d), *(pi1_ri + (i*2 + 1)*dfac + d) ); \
        PASTEMAC(ch,scal2ris)( -kappa_i, kappa_r, *(c_use + 0), *(c_use + 1), \
                                          *(pi1_ir + (i*2 + 0)*dfac + d), *(pi1_ir + (i*2 + 1)*dfac + d) ); \
      }\
      else {\
        PASTEMAC(ch,axpyris)(  kappa_r, kappa_i, *(c_use + 0), *(c_use + 1), \
                                          *(pi1_ri + (i*2 + 0)*dfac + d), *(pi1_ri + (i*2 + 1)*dfac + d) ); \
        PASTEMAC(ch,axpyris)( -kappa_i, kappa_r, *(c_use + 0), *(c_use + 1), \
                                          *(pi1_ir + (i*2 + 0)*dfac + d), *(pi1_ir + (i*2 + 1)*dfac + d) ); \
      }\
      \
    }\
\
    c1 += ldc2; \
    pi1_ri += ldp2; \
    pi1_ir += ldp2; \
  } \
} while(0)


#define PACKM_1R_BODY( ctype, ch, pragma, cdim, dfac, inca2, op ) \
\
do \
{ \
  for ( dim_t k = 0; k < panel_len; k++ ) \
  { \
    pragma \
    for ( dim_t i = 0; i < cdim; i++ ) \
    for ( dim_t d = 0; d < dfac; d++ ) \
      dim_t panel_dim_total_off = panel_dim + i;\
      dim_t panel_len_total_off = panel_len + k;\
      doff_t diagoffc = panel_dim_total_off - panel_len_total_off;\
      ctype* c_use = (c1 + i * incc2); \
      if ( bli_is_triangular( strucc )) {\
          if (diagoffc < 0 && bli_is_upper( uploc ) || \
              diagoffc > 0 && bli_is_lower( uploc )) {\
              c_use = c_begin + panel_len_total_off * incc2 + panel_dim_total_off * ldc2;\
          }\
      }\
      if (acc == 0) {\
        PASTEMAC(ch,scal2ris)( kappa_r, kappa_i, *(c_use + 0), *(c_use + 1), \
                                         *(pi1_r + i*dfac + d), *(pi1_i + i*dfac + d) ); \
      }\
      else {\
        PASTEMAC(ch,axpyris)( kappa_r, kappa_i, *(c_use + 0), *(c_use + 1), \
                                         *(pi1_r + i*dfac + d), *(pi1_i + i*dfac + d) ); \
      }\
      \
    c1 += ldc2; \
    pi1_r  += ldp2; \
    pi1_i  += ldp2; \
  } \
} while(0)


#undef  GENTFUNCCO
#define GENTFUNCCO( ctype, ctype_r, ch, chr, opname, acc_ ) \
\
void PASTEMAC3(ch,opname,acc) \
     ( \
             conj_t  conja, \
             pack_t  schema, \
             dim_t   panel_dim, \
             dim_t   panel_dim_max, \
             dim_t   panel_bcast, \
             dim_t   panel_len, \
             dim_t   panel_len_max, \
       const void*   kappa, \
       const void*   c, inc_t incc, inc_t ldc, \
             void*   p,             inc_t ldp, \
       const void*   params, \
       const cntx_t* cntx  \
     ) \
{ \
  int acc = acc_;\
  const ctype* c_begin = ((ctype*)c) - (panel_dim_off * incc) - (panel_len_off * ldc);\
  const dim_t mr  = PASTECH(BLIS_MR_, ch); \
  const dim_t nr  = PASTECH(BLIS_NR_, ch); \
  const dim_t bbm = PASTECH(BLIS_BBM_, ch); \
  const dim_t bbn = PASTECH(BLIS_BBN_, ch); \
\
  if ( bli_is_1e_packed( schema ) ) \
  { \
    const dim_t panel_dim2 = 2 * panel_dim; \
    const inc_t incc2 = 2 * incc; \
    const inc_t ldc2  = 2 * ldc; \
    const inc_t ldp2  = 2 * ldp; \
\
          ctype_r           kappa_r = ( ( ctype_r* )kappa )[0]; \
          ctype_r           kappa_i = ( ( ctype_r* )kappa )[1]; \
    const ctype_r* restrict c1  = ( ctype_r* )c; \
          ctype_r* restrict pi1_ri  = ( ctype_r* )p; \
          ctype_r* restrict pi1_ir  = ( ctype_r* )p + ldp; \
\
    PACKM_1E_BODY( ctype, ch, PRAGMA_SIMD, panel_dim, panel_bcast, incc2, scal2ris ); \
\
    if (acc == 0) {\
      PASTEMAC(chr,set0s_edge) \
      ( \
        panel_dim2*panel_bcast, 2*panel_dim_max*panel_bcast, \
        2*panel_len, 2*panel_len_max, \
        ( ctype_r* )p, ldp  \
      ); \
    }\
  } \
  else /* ( bli_is_1r_packed( schema ) ) */ \
  { \
    const inc_t incc2 = 2 * incc; \
    const inc_t ldc2  = 2 * ldc; \
    const inc_t ldp2  = 2 * ldp; \
\
          ctype_r           kappa_r = ( ( ctype_r* )kappa )[0]; \
          ctype_r           kappa_i = ( ( ctype_r* )kappa )[1]; \
    const ctype_r* restrict c1  = ( ctype_r* )c; \
          ctype_r* restrict pi1_r   = ( ctype_r* )p; \
          ctype_r* restrict pi1_i   = ( ctype_r* )p + ldp; \
\
    PACKM_1R_BODY( ctype, ch, PRAGMA_SIMD, panel_dim, panel_bcast, incc2, scal2ris ); \
\
    if (acc == 0) {\
      PASTEMAC(chr,set0s_edge) \
      ( \
        panel_dim*panel_bcast, panel_dim_max*panel_bcast, \
        2*panel_len, 2*panel_len_max, \
        ( ctype_r* )p, ldp  \
      ); \
    }\
  } \
}

INSERT_GENTFUNCCO( packm__1er_0, 0 )
INSERT_GENTFUNCCO( packm__1er_1, 1 )

#undef  GENTFUNCR
#define GENTFUNCR( ctype, ctype_r, ch, chr, varname, cxk_kername, cxc_kername, acc_ ) \
\
void PASTEMAC(ch,varname) \
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
     ) \
{ \
    const ctype* c_begin = ((ctype*)c) - (panel_dim_off * incc) - (panel_len_off * ldc);\
    int acc = acc_;\
    cntl_t* cntl          = ( cntl_t* )params; \
    \
    ctype           kappa_cast = *( ctype* )kappa; \
\
    num_t   dt            = PASTEMAC(ch,type); \
    dim_t   panel_len_pad = panel_len_max - panel_len; \
\
    dim_t   packmrnr      = bli_packm_def_cntl_bmult_m_def( cntl ); \
    const ctype* restrict c1     = c; \
    ctype* restrict p1        = p; \
\
    for ( dim_t k = 0; k < panel_len; k++ ) \
    { \
        for ( dim_t i = 0; i < panel_dim; i++ ) {\
            for ( dim_t d = 0; d < panel_bcast; d++ ) {\
                dim_t panel_dim_total_off = panel_dim + i;\
                dim_t panel_len_total_off = panel_len + p;\
                doff_t diagoffc = panel_dim_total_off - panel_len_total_off;\
                ctype* c_use = (c1 + i * incc); \
                if ( bli_is_triangular( strucc )) {\
                    if (diagoffc < 0 && bli_is_upper( uploc ) || \
                        diagoffc > 0 && bli_is_lower( uploc )) {\
                        c_use = c_begin + panel_len_total_off * incc + panel_dim_total_off * ldc;\
                    }\
                }\
                if (acc == 0) {\
                    PASTEMAC(ch, scal2js)( kappa_cast, *(c_use), *(p1 + i*panel_bcast + d) ); \
                }\
                else {\
                    PASTEMAC(ch, axpyjs)( kappa_cast, *(c_use), *(p1 + i*panel_bcast + d) ); \
                }\
            }\
        }\
        \
        c1 += ldc; \
        p1 += ldp; \
    } \
    if (acc == 0) {\
        PASTEMAC(ch,set0s_edge) \
        ( \
          panel_dim*panel_bcast, panel_dim_max*panel_bcast, \
          panel_len, panel_len_max, \
          p1, ldp  \
        ); \
    }\
}

INSERT_GENTFUNCR_BASIC( packm__struc_cxk_0, packm_cxk, packm_cxc_diag, 0 )
INSERT_GENTFUNCR_BASIC( packm__struc_cxk_1, packm_cxk, packm_cxc_diag, 1 )

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
    fmm_params_t*    params    = ( fmm_params_t* )params_; \
    float* restrict coef = ( float* )params->coef; \
    ctype kappa_cast, lambda; \
    kappa_cast = *( ctype* )kappa; \
\
    inc_t* restrict off_m = params->off_m; \
    inc_t* restrict off_k = params->off_n; \
    dim_t* restrict part_m = params->part_m;\
    dim_t* restrict part_n = params->part_n; \
\
    dim_t nsplit = params->nsplit; \
    for (int s = 0; s < nsplit; s++) \
    { \
        if (s == 0)\
        { \
            PASTEMAC3(ch, s, ch, scal2s)( kappa_cast, coef[ s ], lambda );\
            \
            inc_t total_off_m = panel_dim_off + off_m[s];\
            inc_t total_off_n = panel_len_off + off_k[s];\
        \
            const ctype* restrict c_use = ( ctype* )c + off_m[s] * incc + off_k[s] * ldc; \
            ctype* restrict p_use = ( ctype* )p; \
        \
                /* Check if we need to shrink the micro-panel due to unequal partitioning. */ \
            dim_t panel_dim_use = bli_min(panel_dim, part_m[s] - panel_dim_off ); \
            dim_t panel_len_use = bli_min(panel_len, part_n[s] - panel_len_off ); \
            \
            PASTEMAC(ch,packm__struc_cxk_0)(strucc,\
                diagc,\
                uploc,\
                conjc,\
                schema,\
                invdiag,\
                panel_dim_use,\
                panel_len_use,\
                panel_dim_max,\
                panel_len_max,\
                panel_dim_off,\
                panel_len_off,\
                panel_bcast,\
                &lambda,\
                c_use, incc, ldc,\
                p_use, ldp,\
                params_,\
                cntx\
                );\
        } \
        else if (coef[s] != 0) \
        { \
            PASTEMAC3(ch, s, ch, scal2s)( kappa_cast, coef[ s ], lambda );\
            \
            inc_t total_off_m = panel_dim_off + off_m[s];\
            inc_t total_off_n = panel_len_off + off_k[s];\
        \
            const ctype* restrict c_use = ( ctype* )c + off_m[s] * incc + off_k[s] * ldc; \
            ctype* restrict p_use = ( ctype* )p; \
        \
                /* Check if we need to shrink the micro-panel due to unequal partitioning. */ \
            dim_t panel_dim_use = bli_min(panel_dim, part_m[s] - panel_dim_off ); \
            dim_t panel_len_use = bli_min(panel_len, part_n[s] - panel_len_off ); \
            \
            PASTEMAC(ch,packm__struc_cxk_1)(strucc,\
                diagc,\
                uploc,\
                conjc,\
                schema,\
                invdiag,\
                panel_dim_use,\
                panel_len_use,\
                panel_dim_use,\
                panel_len_use,\
                total_off_m,\
                total_off_n,\
                panel_bcast,\
                &lambda,\
                c_use, incc, ldc,\
                p_use, ldp,\
                params_,\
                cntx\
            );\
        }\
    }\
}

            // packm_ker_cast\
            // (\
            //     1,
            //     strucc,\
            //     diagc,\
            //     uploc,\
            //     conjc,\
            //     schema,\
            //     invdiag,\
            //     panel_dim_use,\
            //     panel_len_use,\
            //     panel_dim_use,\
            //     panel_len_use,\
            //     total_off_m,\
            //     total_off_n,\
            //     panel_bcast,\
            //     &lambda,\
            //     c_use, incc, ldc,\
            //     p_use, ldp,\
            //     params_,\
            //     cntx\
            // );\


INSERT_GENTFUNC_BASIC( packm_symm_fmm, BLIS_CNAME_INFIX, BLIS_REF_SUFFIX )
