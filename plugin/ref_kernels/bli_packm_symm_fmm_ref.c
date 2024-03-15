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

// #define PACKM_BODY( ctype, ch, pragma, cdim, dfac, inca, op ) \
// \
// do \
// { \
//     for ( dim_t k = n; k != 0; --k ) \
//     { \
//         pragma \
//         for ( dim_t mn = 0; mn < cdim; mn++ ) \
//         for ( dim_t d = 0; d < dfac; d++ ) \
//             PASTEMAC(ch,op)( kappa_cast, *(alpha1 + mn*inca), *(pi1 + mn*dfac + d) ); \
// \
//         alpha1 += lda; \
//         pi1    += ldp; \
//     } \
// } while(0)


// #undef  GENTFUNC
// #define GENTFUNC( ctype, ch, opname, arch, acc ) \
// \
// void PASTEMAC3(ch,opname,arch,acc) \
//      ( \
//              conj_t  conja, \
//              pack_t  schema, \
//              dim_t   cdim, \
//              dim_t   cdim_max, \
//              dim_t   cdim_bcast, \
//              dim_t   n, \
//              dim_t   n_max, \
//        const void*   kappa, \
//        const void*   a, inc_t inca, inc_t lda, \
//              void*   p,             inc_t ldp, \
//        const void*   params, \
//        const cntx_t* cntx  \
//      ) \
// { \
//     const dim_t mr  = PASTECH(BLIS_MR_, ch); \
//     const dim_t nr  = PASTECH(BLIS_NR_, ch); \
//     const dim_t bbm = PASTECH(BLIS_BBM_, ch); \
//     const dim_t bbn = PASTECH(BLIS_BBN_, ch); \
// \
//           ctype           kappa_cast = *( ctype* )kappa; \
//     const ctype* restrict alpha1     = a; \
//           ctype* restrict pi1        = p; \
//     int acc_ = acc;\
// \
//     if (acc_ == 1) {\
//         if ( cdim == mr && cdim_bcast == bbm && mr != -1 ) \
//         { \
//             if ( inca == 1 ) \
//             { \
//                 if ( bli_is_conj( conja ) ) PACKM_BODY( ctype, ch, PRAGMA_SIMD, mr, bbm, 1, scal2js ); \
//                 else                        PACKM_BODY( ctype, ch, PRAGMA_SIMD, mr, bbm, 1, scal2s ); \
//             } \
//             else \
//             { \
//                 if ( bli_is_conj( conja ) ) PACKM_BODY( ctype, ch, PRAGMA_SIMD, mr, bbm, inca, scal2js ); \
//                 else                        PACKM_BODY( ctype, ch, PRAGMA_SIMD, mr, bbm, inca, scal2s ); \
//             } \
//         } \
//         else if ( cdim == nr && cdim_bcast == bbn && nr != -1 ) \
//         { \
//             if ( inca == 1 ) \
//             { \
//                 if ( bli_is_conj( conja ) ) PACKM_BODY( ctype, ch, PRAGMA_SIMD, nr, bbn, 1, scal2js ); \
//                 else                        PACKM_BODY( ctype, ch, PRAGMA_SIMD, nr, bbn, 1, scal2s ); \
//             } \
//             else \
//             { \
//                 if ( bli_is_conj( conja ) ) PACKM_BODY( ctype, ch, PRAGMA_SIMD, nr, bbn, inca, scal2js ); \
//                 else                        PACKM_BODY( ctype, ch, PRAGMA_SIMD, nr, bbn, inca, scal2s ); \
//             } \
//         } \
//         else \
//         { \
//             if ( bli_is_conj( conja ) ) PACKM_BODY( ctype, ch, , cdim, cdim_bcast, inca, scal2js ); \
//             else                        PACKM_BODY( ctype, ch, , cdim, cdim_bcast, inca, scal2s ); \
//         } \
//     \
//         PASTEMAC(ch,set0s_edge) \
//         ( \
//           cdim*cdim_bcast, cdim_max*cdim_bcast, \
//           n, n_max, \
//           p, ldp  \
//         ); \
//     }\
//     else {\
//         if ( cdim == mr && cdim_bcast == bbm && mr != -1 ) \
//         { \
//             if ( inca == 1 ) \
//             { \
//                 if ( bli_is_conj( conja ) ) PACKM_BODY( ctype, ch, PRAGMA_SIMD, mr, bbm, 1, axpyjs ); \
//                 else                        PACKM_BODY( ctype, ch, PRAGMA_SIMD, mr, bbm, 1, axpys ); \
//             } \
//             else \
//             { \
//                 if ( bli_is_conj( conja ) ) PACKM_BODY( ctype, ch, PRAGMA_SIMD, mr, bbm, inca, axpyjs ); \
//                 else                        PACKM_BODY( ctype, ch, PRAGMA_SIMD, mr, bbm, inca, axpys ); \
//             } \
//         } \
//         else if ( cdim == nr && cdim_bcast == bbn && nr != -1 ) \
//         { \
//             if ( inca == 1 ) \
//             { \
//                 if ( bli_is_conj( conja ) ) PACKM_BODY( ctype, ch, PRAGMA_SIMD, nr, bbn, 1, axpyjs ); \
//                 else                        PACKM_BODY( ctype, ch, PRAGMA_SIMD, nr, bbn, 1, axpys ); \
//             } \
//             else \
//             { \
//                 if ( bli_is_conj( conja ) ) PACKM_BODY( ctype, ch, PRAGMA_SIMD, nr, bbn, inca, axpyjs ); \
//                 else                        PACKM_BODY( ctype, ch, PRAGMA_SIMD, nr, bbn, inca, axpys ); \
//             } \
//         } \
//         else \
//         { \
//             if ( bli_is_conj( conja ) ) PACKM_BODY( ctype, ch, , cdim, cdim_bcast, inca, axpyjs ); \
//             else                        PACKM_BODY( ctype, ch, , cdim, cdim_bcast, inca, axpys ); \
//         } \
//     }\
// }

// INSERT_GENTFUNC_BASIC( pack_m, _fmm_, 0 )
// INSERT_GENTFUNC_BASIC( pack_m, _fmm_, 1 )


// #undef  GENTFUNCR
// #define GENTFUNCR( ctype, ctype_r, ch, chr, varname, cxk_kername, cxc_kername, acc_ ) \
// \
// void PASTEMAC(ch,varname) \
//      ( \
//              struc_t strucc, \
//              diag_t  diagc, \
//              uplo_t  uploc, \
//              conj_t  conjc, \
//              pack_t  schema, \
//              bool    invdiag, \
//              dim_t   panel_dim, \
//              dim_t   panel_len, \
//              dim_t   panel_dim_max, \
//              dim_t   panel_len_max, \
//              dim_t   panel_dim_off, \
//              dim_t   panel_len_off, \
//              dim_t   panel_bcast, \
//        const void*   kappa, \
//        const void*   c, inc_t incc, inc_t ldc, \
//              void*   p,             inc_t ldp, \
//        const void*   params, \
//        const cntx_t* cntx  \
//      ) \
// { \
//     int acc = acc_;\
//     cntl_t* cntl          = ( cntl_t* )params; \
// \
//     num_t   dt            = PASTEMAC(ch,type); \
//     dim_t   panel_len_pad = panel_len_max - panel_len; \
// \
//     dim_t   packmrnr      = bli_packm_def_cntl_bmult_m_def( cntl ); \
// \
//     ukr_t   cxk_ker_id    = BLIS_PACKM_KER; \
//     ukr_t   cxc_ker_id    = BLIS_PACKM_DIAG_KER; \
// \
//     if ( bli_is_1m_packed( schema ) ) \
//     { \
//         cxk_ker_id = BLIS_PACKM_1ER_KER; \
//         cxc_ker_id = BLIS_PACKM_DIAG_1ER_KER; \
//     } \
// \
//     PASTECH(cxk_kername,_ker_ft) f_cxk;\
//     if (acc == 0)\
//         f_cxk = PASTEMAC3(ch,pack_m,_fmm_,0);\
//     else\
//         f_cxk = PASTEMAC3(ch,pack_m,_fmm_,1);\
// \
//     PASTECH(cxc_kername,_ker_ft) f_cxc = bli_cntx_get_ukr_dt( dt, cxc_ker_id, cntx ); \
// \
//     /* For general matrices, pack and return early */ \
//     if ( bli_is_general( strucc ) ) \
//     { \
//         f_cxk \
//         ( \
//           conjc, \
//           schema, \
//           panel_dim, \
//           panel_dim_max, \
//           panel_bcast, \
//           panel_len, \
//           panel_len_max, \
//           kappa, \
//           c, incc, ldc, \
//           p,       ldp, \
//           params, \
//           cntx  \
//         ); \
//         return; \
//     } \
// \
//     /* Sanity check. Diagonals should not intersect the short end of
//        a micro-panel. If they do, then somehow the constraints on
//        cache blocksizes being a whole multiple of the register
//        blocksizes was somehow violated. */ \
//     doff_t diagoffc = panel_dim_off - panel_len_off; \
//     if ( (          -panel_dim < diagoffc && diagoffc <         0 ) || \
//          ( panel_len-panel_dim < diagoffc && diagoffc < panel_len ) ) \
//     {\
//         printf("%d %d %d\n", panel_dim_off, panel_len_off, diagoffc);\
//         printf("%d %d\n", panel_len, panel_dim);\
//         bli_check_error_code( BLIS_NOT_YET_IMPLEMENTED ); \
//     }\
// \
//     /* For triangular, symmetric, and hermitian matrices we need to consider
//        three parts. */ \
// \
//     /* Pack to p10. */ \
//     if ( 0 < diagoffc ) \
//     { \
//         dim_t  p10_len     = bli_min( diagoffc, panel_len ); \
//         dim_t  p10_len_max = p10_len == panel_len ? panel_len_max : p10_len; \
//         ctype* p10         = ( ctype* )p; \
//         conj_t conjc10     = conjc; \
//         ctype* c10         = ( ctype* )c; \
//         inc_t  incc10      = incc; \
//         inc_t  ldc10       = ldc; \
// \
//         if ( bli_is_upper( uploc ) ) \
//         { \
//             bli_reflect_to_stored_part( diagoffc, c10, incc10, ldc10 ); \
// \
//             if ( bli_is_hermitian( strucc ) ) \
//                 bli_toggle_conj( &conjc10 ); \
//         } \
// \
//         /* If we are referencing the unstored part of a triangular matrix,
//            explicitly store zeros */ \
//         if ( bli_is_upper( uploc ) && bli_is_triangular( strucc ) ) \
//         { \
//             if ( bli_is_1m_packed( schema ) ) \
//             { \
//                 ctype_r* restrict zero = PASTEMAC(chr,0); \
// \
//                 PASTEMAC2(chr,setm,BLIS_TAPI_EX_SUF) \
//                 ( \
//                   BLIS_NO_CONJUGATE, \
//                   0, \
//                   BLIS_NONUNIT_DIAG, \
//                   BLIS_DENSE, \
//                   packmrnr, \
//                   p10_len_max * 2, \
//                   zero, \
//                   ( ctype_r* )p10, 1, ldp, \
//                   cntx, \
//                   NULL  \
//                 ); \
//             } \
//             else \
//             { \
//                 ctype* restrict zero = PASTEMAC(ch,0); \
// \
//                 PASTEMAC2(ch,setm,BLIS_TAPI_EX_SUF) \
//                 ( \
//                   BLIS_NO_CONJUGATE, \
//                   0, \
//                   BLIS_NONUNIT_DIAG, \
//                   BLIS_DENSE, \
//                   packmrnr, \
//                   p10_len_max, \
//                   zero, \
//                   p10, 1, ldp, \
//                   cntx, \
//                   NULL  \
//                 ); \
//             } \
//         } \
//         else \
//         { \
//             f_cxk \
//             ( \
//               conjc10, \
//               schema, \
//               panel_dim, \
//               panel_dim_max, \
//               panel_bcast, \
//               p10_len, \
//               p10_len_max, \
//               kappa, \
//               c10, incc10, ldc10, \
//               p10,         ldp, \
//               params, \
//               cntx  \
//             ); \
//         } \
//     } \
// \
//     /* Pack to p11. */ \
//     if ( 0 <= diagoffc && diagoffc + panel_dim <= panel_len ) \
//     { \
//         dim_t  i           = diagoffc; \
//         dim_t  p11_len_max = panel_dim + ( diagoffc + panel_dim == panel_len \
//                                            ? panel_len_pad : 0 ); \
//         ctype* p11         = ( ctype* )p + i * ldp; \
//         conj_t conjc11     = conjc; \
//         ctype* c11         = ( ctype* )c + i * ldc; \
//         inc_t  incc11      = incc; \
//         inc_t  ldc11       = ldc; \
// \
//         f_cxc \
//         ( \
//           strucc, \
//           diagc, \
//           uploc, \
//           conjc11, \
//           schema, \
//           invdiag, \
//           panel_dim, \
//           panel_dim_max, \
//           panel_bcast, \
//           p11_len_max, \
//           kappa, \
//           c11, incc11, ldc11, \
//           p11,         ldp, \
//           params, \
//           cntx  \
//         ); \
//     } \
// \
//     /* Pack to p12. */ \
//     if ( diagoffc + panel_dim < panel_len ) \
//     { \
//         dim_t  i           = bli_max( 0, diagoffc + panel_dim ); \
//         dim_t  p12_len     = panel_len - i; \
//         /* If we are packing p12, then it is always the last partial block \
//            and so we should make sure to pad with zeros if necessary. */ \
//         dim_t  p12_len_max = p12_len + panel_len_pad; \
//         ctype* p12         = ( ctype* )p + i * ldp; \
//         conj_t conjc12     = conjc; \
//         ctype* c12         = ( ctype* )c + i * ldc; \
//         inc_t  incc12      = incc; \
//         inc_t  ldc12       = ldc; \
// \
//         if ( bli_is_lower( uploc ) ) \
//         { \
//             bli_reflect_to_stored_part( diagoffc - i, c12, incc12, ldc12 ); \
// \
//             if ( bli_is_hermitian( strucc ) ) \
//                 bli_toggle_conj( &conjc12 ); \
//         } \
// \
//         /* If we are referencing the unstored part of a triangular matrix,
//            explicitly store zeros */ \
//         if ( bli_is_lower( uploc ) && bli_is_triangular( strucc ) ) \
//         { \
//             if ( 0) \
//             { \
//                 \
//             } \
//             else \
//             { \
//                 ctype* restrict zero = PASTEMAC(ch,0); \
// \
//                 PASTEMAC2(ch,setm,BLIS_TAPI_EX_SUF) \
//                 ( \
//                   BLIS_NO_CONJUGATE, \
//                   0, \
//                   BLIS_NONUNIT_DIAG, \
//                   BLIS_DENSE, \
//                   packmrnr, \
//                   p12_len_max, \
//                   zero, \
//                   p12, 1, ldp, \
//                   cntx, \
//                   NULL  \
//                 ); \
//             } \
//         } \
//         else \
//         { \
//             f_cxk \
//             ( \
//               conjc12, \
//               schema, \
//               panel_dim, \
//               panel_dim_max, \
//               panel_bcast, \
//               p12_len, \
//               p12_len_max, \
//               kappa, \
//               c12, incc12, ldc12, \
//               p12,         ldp, \
//               params, \
//               cntx  \
//             ); \
//         } \
//     } \
// }

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

// for ( dim_t p = 0; p < panel_len; p++ ) \
//     { \
//         for ( dim_t i = 0; i < panel_dim; i++ ) {\
//             for ( dim_t d = 0; d < panel_bcast; d++ ) {\
//                 dim_t panel_dim_total_off = panel_dim + i;\
//                 dim_t panel_len_total_off = panel_len + p;\
//                 doff_t diagoffc = panel_dim_total_off - panel_len_total_off;\
//                 void* c_use = (c + i * incc); \
//                 if ( bli_is_triangular( strucc ) ) {\
//                     if (diagoffc < 0 && bli_is_upper( uploc ) || \
//                         diagoffc > 0 && bli_is_lower( uploc )) {\
//                         c_use = c_begin + panel_len_total_off * incc + panel_dim_total_off * ldc;\
//                     }\
//                 }\
//                 if (acc == 0) {\
//                     PASTEMAC(ch, scal2js)( kappa_cast, *(c_use), *(p + i*panel_bcast + d) ); \
//                 }\
//                 else {\
//                     PASTEMAC(ch, axpyjs)( kappa_cast, *(c_use), *(p + i*panel_bcast + d) ); \
//                 }\
//             }\
//         }\
//         \
//         c += ldc; \
//         p += ldp; \
//     } \
//     if (acc == 0) {\
//         PASTEMAC(ch,set0s_edge) \
//         ( \
//           panel_dim*panel_bcast, panel_dim_max*panel_bcast, \
//           panel_len, panel_len_max, \
//           p, ldp  \
//         ); \
//     }\

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
