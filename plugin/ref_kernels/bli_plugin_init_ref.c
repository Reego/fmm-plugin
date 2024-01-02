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

// -- Macros to help concisely instantiate bli_func_init() ---------------------

#define gen_func_init_ro( func_p, opname ) \
do { \
	bli_func_init( func_p, PASTEMAC(s,opname), PASTEMAC(d,opname), \
	                       NULL,               NULL ); \
} while (0)

#define gen_func_init_co( func_p, opname ) \
do { \
	bli_func_init( func_p, NULL,               NULL, \
	                       PASTEMAC(c,opname), PASTEMAC(z,opname) ); \
} while (0)

#define gen_func_init( func_p, opname ) \
do { \
	bli_func_init( func_p, PASTEMAC(s,opname), PASTEMAC(d,opname), \
	                       PASTEMAC(c,opname), PASTEMAC(z,opname) ); \
} while (0)

// -----------------------------------------------------------------------------

void PASTEMAC3(plugin_init,BLIS_PNAME_INFIX,BLIS_CNAME_INFIX,BLIS_REF_SUFFIX)( PASTECH2(plugin,BLIS_PNAME_INFIX,_params) )
{
	cntx_t* cntx = ( cntx_t* )bli_gks_lookup_id( PASTECH(BLIS_ARCH,BLIS_CNAME_UPPER_INFIX) );

	func_t pack_ukr, gemm_ukr;
	gen_func_init( &pack_ukr, PASTECH2(packm_fmm,BLIS_CNAME_INFIX,BLIS_REF_SUFFIX) );
	gen_func_init( &gemm_ukr, PASTECH2(gemm_fmm,BLIS_CNAME_INFIX,BLIS_REF_SUFFIX) );

	bli_cntx_set_ukr( FMM_BLIS_PACK_UKR, &pack_ukr, cntx );
	bli_cntx_set_ukr( FMM_BLIS_GEMM_UKR, &gemm_ukr, cntx );
}

