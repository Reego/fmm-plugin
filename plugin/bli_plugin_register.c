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
#include "bli_plugin_fmm_blis.h"

#include STRINGIFY_INT(PASTEMAC(plugin,BLIS_PNAME_INFIX).h)

siz_t FMM_BLIS_PACK_UKR;
siz_t FMM_BLIS_PACK_UKR_SYMM;
siz_t FMM_BLIS_GEMM_UKR;
siz_t FMM_BLIS_GEMM1M_UKR;

err_t PASTEMAC(plugin_register,BLIS_PNAME_INFIX)( PASTECH2(plugin,BLIS_PNAME_INFIX,_params) )
{
	err_t err;

	err = bli_gks_register_ukr(&FMM_BLIS_PACK_UKR);
  err = bli_gks_register_ukr(&FMM_BLIS_PACK_UKR_SYMM);
	err = bli_gks_register_ukr(&FMM_BLIS_GEMM_UKR);
  err = bli_gks_register_ukr(&FMM_BLIS_GEMM1M_UKR);

	if ( err != BLIS_SUCCESS )
		return err;

	//
	// Initialize the context for each enabled sub-configuration.
	//

	#undef GENTCONF
	#define GENTCONF( CONFIG, config ) \
	PASTEMAC4(plugin_init,BLIS_PNAME_INFIX,_,config,BLIS_REF_SUFFIX)( PASTECH2(plugin,BLIS_PNAME_INFIX,_params_only) );

	INSERT_GENTCONF

	return err;
}

double max_diff(obj_t* A, obj_t* B)
{

    const double TOLERANCE = 1E-10;

    dim_t m = A->dim[0];
    dim_t n = A->dim[1];

    inc_t rsa = A->rs;
    inc_t csa = A->cs;

    inc_t rsb = B->rs;
    inc_t csb = B->cs;

    int    i, j;
    double max = 0;
    for ( i = 0; i < m; i ++ ) {
        for ( j = 0; j < n; j ++ ) {
            double a_element = ((double*)A->buffer)[i * rsa + j * csa];
            double b_element = ((double*)B->buffer)[i * rsb + j * csb];

            if (fabs(a_element - b_element) > max)
                max = fabs(a_element - b_element);
        }
    }
    return max;
}

