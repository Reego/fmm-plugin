
#include "bl_dgemm.h"

void bl_dgemm_ref(
        int    m,
        int    n,
        int    k,
        double *XA,
        int    lda,
        double *XB,
        int    ldb,
        double *XC,
        int    ldc
        )
{
    // Local variables.
    int    i, j, p;
    double alpha = 1.0, beta = 1.0;

    // Sanity check for early return.
    if ( m == 0 || n == 0 || k == 0 ) return;

    // Reference GEMM implementation.

    dgemm_( "N", "N", &m, &n, &k, &alpha,
            XA, &lda, XB, &ldb, &beta, XC, &ldc );

}

