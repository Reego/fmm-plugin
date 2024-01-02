#include "bl_dgemm_kernel.h"
#include "bl_dgemm.h"
#include <time.h>

void M_Add0( int m, int n, double* M0, double* M1, int ldM, double* R, int ldR, int bl_ic_nt ) {
    int i, j;
#ifdef _PARALLEL_
    #pragma omp parallel for num_threads( bl_ic_nt )
#endif
    for ( j = 0; j < n; ++j ) {
        for ( i = 0; i < m; ++i ) {
            M0[ i + j * ldM ] +=  + R[ i + j * ldR ];
            M1[ i + j * ldM ] +=  + R[ i + j * ldR ];
        }
    }
}

void M_Add1( int m, int n, double* M0, double* M1, int ldM, double* R, int ldR, int bl_ic_nt ) {
    int i, j;
#ifdef _PARALLEL_
    #pragma omp parallel for num_threads( bl_ic_nt )
#endif
    for ( j = 0; j < n; ++j ) {
        for ( i = 0; i < m; ++i ) {
            M0[ i + j * ldM ] +=  + R[ i + j * ldR ];
            M1[ i + j * ldM ] +=  -R[ i + j * ldR ];
        }
    }
}

void M_Add2( int m, int n, double* M0, double* M1, int ldM, double* R, int ldR, int bl_ic_nt ) {
    int i, j;
#ifdef _PARALLEL_
    #pragma omp parallel for num_threads( bl_ic_nt )
#endif
    for ( j = 0; j < n; ++j ) {
        for ( i = 0; i < m; ++i ) {
            M0[ i + j * ldM ] +=  + R[ i + j * ldR ];
            M1[ i + j * ldM ] +=  + R[ i + j * ldR ];
        }
    }
}

void M_Add3( int m, int n, double* M0, double* M1, int ldM, double* R, int ldR, int bl_ic_nt ) {
    int i, j;
#ifdef _PARALLEL_
    #pragma omp parallel for num_threads( bl_ic_nt )
#endif
    for ( j = 0; j < n; ++j ) {
        for ( i = 0; i < m; ++i ) {
            M0[ i + j * ldM ] +=  + R[ i + j * ldR ];
            M1[ i + j * ldM ] +=  + R[ i + j * ldR ];
        }
    }
}

void M_Add4( int m, int n, double* M0, double* M1, int ldM, double* R, int ldR, int bl_ic_nt ) {
    int i, j;
#ifdef _PARALLEL_
    #pragma omp parallel for num_threads( bl_ic_nt )
#endif
    for ( j = 0; j < n; ++j ) {
        for ( i = 0; i < m; ++i ) {
            M0[ i + j * ldM ] +=  -R[ i + j * ldR ];
            M1[ i + j * ldM ] +=  + R[ i + j * ldR ];
        }
    }
}

void M_Add5( int m, int n, double* M0, int ldM, double* R, int ldR, int bl_ic_nt ) {
    int i, j;
#ifdef _PARALLEL_
    #pragma omp parallel for num_threads( bl_ic_nt )
#endif
    for ( j = 0; j < n; ++j ) {
        for ( i = 0; i < m; ++i ) {
            M0[ i + j * ldM ] +=  + R[ i + j * ldR ];
        }
    }
}

void M_Add6( int m, int n, double* M0, int ldM, double* R, int ldR, int bl_ic_nt ) {
    int i, j;
#ifdef _PARALLEL_
    #pragma omp parallel for num_threads( bl_ic_nt )
#endif
    for ( j = 0; j < n; ++j ) {
        for ( i = 0; i < m; ++i ) {
            M0[ i + j * ldM ] +=  + R[ i + j * ldR ];
        }
    }
}

static inline void packA_add_stra_ab0( int m, int n, double *A0, double *A1, int ldA, double *packA ) {
    int i, j;
    double *A0_pntr, *A1_pntr, *packA_pntr;
    for ( j = 0; j < n; ++j ) {
        packA_pntr = &packA[ DGEMM_MR * j ];
        A0_pntr = &A0[ ldA * j ]; A1_pntr = &A1[ ldA * j ]; 
        for ( i = 0; i < DGEMM_MR; ++i ) {
            packA_pntr[ i ] = A0_pntr[ i ] + A1_pntr[ i ];
        }
    }
}

static inline void packA_add_stra_ab1( int m, int n, double *A0, double *A1, int ldA, double *packA ) {
    int i, j;
    double *A0_pntr, *A1_pntr, *packA_pntr;
    for ( j = 0; j < n; ++j ) {
        packA_pntr = &packA[ DGEMM_MR * j ];
        A0_pntr = &A0[ ldA * j ]; A1_pntr = &A1[ ldA * j ]; 
        for ( i = 0; i < DGEMM_MR; ++i ) {
            packA_pntr[ i ] = A0_pntr[ i ] + A1_pntr[ i ];
        }
    }
}

static inline void packA_add_stra_ab2( int m, int n, double *A0, int ldA, double *packA ) {
    int i, j;
    double *A0_pntr, *packA_pntr;
    for ( j = 0; j < n; ++j ) {
        packA_pntr = &packA[ DGEMM_MR * j ];
        A0_pntr = &A0[ ldA * j ]; 
        for ( i = 0; i < DGEMM_MR; ++i ) {
            packA_pntr[ i ] = A0_pntr[ i ];
        }
    }
}

static inline void packA_add_stra_ab3( int m, int n, double *A0, int ldA, double *packA ) {
    int i, j;
    double *A0_pntr, *packA_pntr;
    for ( j = 0; j < n; ++j ) {
        packA_pntr = &packA[ DGEMM_MR * j ];
        A0_pntr = &A0[ ldA * j ]; 
        for ( i = 0; i < DGEMM_MR; ++i ) {
            packA_pntr[ i ] = A0_pntr[ i ];
        }
    }
}

static inline void packA_add_stra_ab4( int m, int n, double *A0, double *A1, int ldA, double *packA ) {
    int i, j;
    double *A0_pntr, *A1_pntr, *packA_pntr;
    for ( j = 0; j < n; ++j ) {
        packA_pntr = &packA[ DGEMM_MR * j ];
        A0_pntr = &A0[ ldA * j ]; A1_pntr = &A1[ ldA * j ]; 
        for ( i = 0; i < DGEMM_MR; ++i ) {
            packA_pntr[ i ] = A0_pntr[ i ] + A1_pntr[ i ];
        }
    }
}

static inline void packA_add_stra_ab5( int m, int n, double *A0, double *A1, int ldA, double *packA ) {
    int i, j;
    double *A0_pntr, *A1_pntr, *packA_pntr;
    for ( j = 0; j < n; ++j ) {
        packA_pntr = &packA[ DGEMM_MR * j ];
        A0_pntr = &A0[ ldA * j ]; A1_pntr = &A1[ ldA * j ]; 
        for ( i = 0; i < DGEMM_MR; ++i ) {
            packA_pntr[ i ] = - A0_pntr[ i ] + A1_pntr[ i ];
        }
    }
}

static inline void packA_add_stra_ab6( int m, int n, double *A0, double *A1, int ldA, double *packA ) {
    int i, j;
    double *A0_pntr, *A1_pntr, *packA_pntr;
    for ( j = 0; j < n; ++j ) {
        packA_pntr = &packA[ DGEMM_MR * j ];
        A0_pntr = &A0[ ldA * j ]; A1_pntr = &A1[ ldA * j ]; 
        for ( i = 0; i < DGEMM_MR; ++i ) {
            packA_pntr[ i ] = A0_pntr[ i ] - A1_pntr[ i ];
        }
    }
}

static inline void packB_add_stra_ab0( int m, int n, double *B0, double *B1, int ldB, double *packB ) {
    int i, j;
    double *B0_pntr, *B1_pntr, *packB_pntr;
    for ( j = 0; j < n; ++j ) {
        packB_pntr = &packB[ DGEMM_NR * j ];
        B0_pntr = &B0[ j ]; B1_pntr = &B1[ j ]; 
        for ( i = 0; i < DGEMM_NR; ++i ) {
            packB_pntr[ i ] = B0_pntr[ i * ldB ] + B1_pntr[ i * ldB ];
        }
    }
}

static inline void packB_add_stra_ab1( int m, int n, double *B0, int ldB, double *packB ) {
    int i, j;
    double *B0_pntr, *packB_pntr;
    for ( j = 0; j < n; ++j ) {
        packB_pntr = &packB[ DGEMM_NR * j ];
        B0_pntr = &B0[ j ]; 
        for ( i = 0; i < DGEMM_NR; ++i ) {
            packB_pntr[ i ] = B0_pntr[ i * ldB ];
        }
    }
}

static inline void packB_add_stra_ab2( int m, int n, double *B0, double *B1, int ldB, double *packB ) {
    int i, j;
    double *B0_pntr, *B1_pntr, *packB_pntr;
    for ( j = 0; j < n; ++j ) {
        packB_pntr = &packB[ DGEMM_NR * j ];
        B0_pntr = &B0[ j ]; B1_pntr = &B1[ j ]; 
        for ( i = 0; i < DGEMM_NR; ++i ) {
            packB_pntr[ i ] = B0_pntr[ i * ldB ] - B1_pntr[ i * ldB ];
        }
    }
}

static inline void packB_add_stra_ab3( int m, int n, double *B0, double *B1, int ldB, double *packB ) {
    int i, j;
    double *B0_pntr, *B1_pntr, *packB_pntr;
    for ( j = 0; j < n; ++j ) {
        packB_pntr = &packB[ DGEMM_NR * j ];
        B0_pntr = &B0[ j ]; B1_pntr = &B1[ j ]; 
        for ( i = 0; i < DGEMM_NR; ++i ) {
            packB_pntr[ i ] = - B0_pntr[ i * ldB ] + B1_pntr[ i * ldB ];
        }
    }
}

static inline void packB_add_stra_ab4( int m, int n, double *B0, int ldB, double *packB ) {
    int i, j;
    double *B0_pntr, *packB_pntr;
    for ( j = 0; j < n; ++j ) {
        packB_pntr = &packB[ DGEMM_NR * j ];
        B0_pntr = &B0[ j ]; 
        for ( i = 0; i < DGEMM_NR; ++i ) {
            packB_pntr[ i ] = B0_pntr[ i * ldB ];
        }
    }
}

static inline void packB_add_stra_ab5( int m, int n, double *B0, double *B1, int ldB, double *packB ) {
    int i, j;
    double *B0_pntr, *B1_pntr, *packB_pntr;
    for ( j = 0; j < n; ++j ) {
        packB_pntr = &packB[ DGEMM_NR * j ];
        B0_pntr = &B0[ j ]; B1_pntr = &B1[ j ]; 
        for ( i = 0; i < DGEMM_NR; ++i ) {
            packB_pntr[ i ] = B0_pntr[ i * ldB ] + B1_pntr[ i * ldB ];
        }
    }
}

static inline void packB_add_stra_ab6( int m, int n, double *B0, double *B1, int ldB, double *packB ) {
    int i, j;
    double *B0_pntr, *B1_pntr, *packB_pntr;
    for ( j = 0; j < n; ++j ) {
        packB_pntr = &packB[ DGEMM_NR * j ];
        B0_pntr = &B0[ j ]; B1_pntr = &B1[ j ]; 
        for ( i = 0; i < DGEMM_NR; ++i ) {
            packB_pntr[ i ] = B0_pntr[ i * ldB ] + B1_pntr[ i * ldB ];
        }
    }
}

static inline void bl_macro_kernel_stra_ab( int m, int n, int k, double *packA, double *packB, double *C, int ldC ) {
    int i, j;
    aux_t aux;
    aux.b_next = packB;
    for ( j = 0; j < n; j += DGEMM_NR ) {
        aux.n  = min( n - j, DGEMM_NR );
        for ( i = 0; i < m; i += DGEMM_MR ) {
            aux.m = min( m - i, DGEMM_MR );
            if ( i + DGEMM_MR >= m ) {
                aux.b_next += DGEMM_NR * k;
            }
            ( *bl_micro_kernel )( k, &packA[ i * k ], &packB[ j * k ], &C[ j * ldC + i ], (unsigned long long) ldC, &aux );
        }
    }
}
// M0 = (1 * a_0 + 1 * a_3) * (1 * b_0 + 1 * b_3);  c_0 += 1 * M0;  c_3 += 1 * M0;
void bl_dgemm_straprim_ab0( int m, int n, int k, double* a0, double* a1, int lda, double* b0, double* b1, int ldb, double* c0, double* c1, int ldc, double *packA, double *packB, int bl_ic_nt ) {
    int i, j, p, ic, ib, jc, jb, pc, pb;
    int ldM = m, nM = n;
    double *M = bl_malloc_aligned( ldM, nM, sizeof(double) );
    memset( M, 0, sizeof(double) * ldM * nM );
    for ( jc = 0; jc < n; jc += DGEMM_NC ) {
        jb = min( n - jc, DGEMM_NC );
        for ( pc = 0; pc < k; pc += DGEMM_KC ) {
            pb = min( k - pc, DGEMM_KC );
#ifdef _PARALLEL_
            #pragma omp parallel for num_threads( bl_ic_nt ) private( j )
#endif
            for ( j = 0; j < jb; j += DGEMM_NR ) {
                packB_add_stra_ab0( min( jb - j, DGEMM_NR ), pb, &b0[ pc + (jc+j)*ldb ], &b1[ pc + (jc+j)*ldb ], ldb, &packB[ j * pb ] );
            }
#ifdef _PARALLEL_
            #pragma omp parallel num_threads( bl_ic_nt ) private( ic, ib, i )
#endif
            {
                int tid = omp_get_thread_num();
                int my_start;
                int my_end;
                bl_get_range( m, DGEMM_MR, &my_start, &my_end );
                for ( ic = my_start; ic < my_end; ic += DGEMM_MC ) {
                    ib = min( my_end - ic, DGEMM_MC );
                    for ( i = 0; i < ib; i += DGEMM_MR ) {
                        packA_add_stra_ab0( min( ib - i, DGEMM_MR ), pb, &a0[ pc*lda + (ic+i) ], &a1[ pc*lda + (ic+i) ], lda, &packA[ tid * DGEMM_MC * pb + i * pb ] );
                    }
                    bl_macro_kernel_stra_ab( ib, jb, pb, packA + tid * DGEMM_MC * pb, packB, &M[ jc * ldM + ic ], ldM );
                }
            }
        }
    }
    M_Add0( m, n, c0, c1, ldc, M, ldM, bl_ic_nt );
    free( M );
}

// M1 = (1 * a_2 + 1 * a_3) * (1 * b_0);  c_2 += 1 * M1;  c_3 += -1 * M1;
void bl_dgemm_straprim_ab1( int m, int n, int k, double* a0, double* a1, int lda, double* b0, int ldb, double* c0, double* c1, int ldc, double *packA, double *packB, int bl_ic_nt ) {
    int i, j, p, ic, ib, jc, jb, pc, pb;
    int ldM = m, nM = n;
    double *M = bl_malloc_aligned( ldM, nM, sizeof(double) );
    memset( M, 0, sizeof(double) * ldM * nM );
    for ( jc = 0; jc < n; jc += DGEMM_NC ) {
        jb = min( n - jc, DGEMM_NC );
        for ( pc = 0; pc < k; pc += DGEMM_KC ) {
            pb = min( k - pc, DGEMM_KC );
#ifdef _PARALLEL_
            #pragma omp parallel for num_threads( bl_ic_nt ) private( j )
#endif
            for ( j = 0; j < jb; j += DGEMM_NR ) {
                packB_add_stra_ab1( min( jb - j, DGEMM_NR ), pb, &b0[ pc + (jc+j)*ldb ], ldb, &packB[ j * pb ] );
            }
#ifdef _PARALLEL_
            #pragma omp parallel num_threads( bl_ic_nt ) private( ic, ib, i )
#endif
            {
                int tid = omp_get_thread_num();
                int my_start;
                int my_end;
                bl_get_range( m, DGEMM_MR, &my_start, &my_end );
                for ( ic = my_start; ic < my_end; ic += DGEMM_MC ) {
                    ib = min( my_end - ic, DGEMM_MC );
                    for ( i = 0; i < ib; i += DGEMM_MR ) {
                        packA_add_stra_ab1( min( ib - i, DGEMM_MR ), pb, &a0[ pc*lda + (ic+i) ], &a1[ pc*lda + (ic+i) ], lda, &packA[ tid * DGEMM_MC * pb + i * pb ] );
                    }
                    bl_macro_kernel_stra_ab( ib, jb, pb, packA + tid * DGEMM_MC * pb, packB, &M[ jc * ldM + ic ], ldM );
                }
            }
        }
    }
    M_Add1( m, n, c0, c1, ldc, M, ldM, bl_ic_nt );
    free( M );
}

// M2 = (1 * a_0) * (1 * b_1 + -1 * b_3);  c_1 += 1 * M2;  c_3 += 1 * M2;
void bl_dgemm_straprim_ab2( int m, int n, int k, double* a0, int lda, double* b0, double* b1, int ldb, double* c0, double* c1, int ldc, double *packA, double *packB, int bl_ic_nt ) {
    int i, j, p, ic, ib, jc, jb, pc, pb;
    int ldM = m, nM = n;
    double *M = bl_malloc_aligned( ldM, nM, sizeof(double) );
    memset( M, 0, sizeof(double) * ldM * nM );
    for ( jc = 0; jc < n; jc += DGEMM_NC ) {
        jb = min( n - jc, DGEMM_NC );
        for ( pc = 0; pc < k; pc += DGEMM_KC ) {
            pb = min( k - pc, DGEMM_KC );
#ifdef _PARALLEL_
            #pragma omp parallel for num_threads( bl_ic_nt ) private( j )
#endif
            for ( j = 0; j < jb; j += DGEMM_NR ) {
                packB_add_stra_ab2( min( jb - j, DGEMM_NR ), pb, &b0[ pc + (jc+j)*ldb ], &b1[ pc + (jc+j)*ldb ], ldb, &packB[ j * pb ] );
            }
#ifdef _PARALLEL_
            #pragma omp parallel num_threads( bl_ic_nt ) private( ic, ib, i )
#endif
            {
                int tid = omp_get_thread_num();
                int my_start;
                int my_end;
                bl_get_range( m, DGEMM_MR, &my_start, &my_end );
                for ( ic = my_start; ic < my_end; ic += DGEMM_MC ) {
                    ib = min( my_end - ic, DGEMM_MC );
                    for ( i = 0; i < ib; i += DGEMM_MR ) {
                        packA_add_stra_ab2( min( ib - i, DGEMM_MR ), pb, &a0[ pc*lda + (ic+i) ], lda, &packA[ tid * DGEMM_MC * pb + i * pb ] );
                    }
                    bl_macro_kernel_stra_ab( ib, jb, pb, packA + tid * DGEMM_MC * pb, packB, &M[ jc * ldM + ic ], ldM );
                }
            }
        }
    }
    M_Add2( m, n, c0, c1, ldc, M, ldM, bl_ic_nt );
    free( M );
}

// M3 = (1 * a_3) * (-1 * b_0 + 1 * b_2);  c_0 += 1 * M3;  c_2 += 1 * M3;
void bl_dgemm_straprim_ab3( int m, int n, int k, double* a0, int lda, double* b0, double* b1, int ldb, double* c0, double* c1, int ldc, double *packA, double *packB, int bl_ic_nt ) {
    int i, j, p, ic, ib, jc, jb, pc, pb;
    int ldM = m, nM = n;
    double *M = bl_malloc_aligned( ldM, nM, sizeof(double) );
    memset( M, 0, sizeof(double) * ldM * nM );
    for ( jc = 0; jc < n; jc += DGEMM_NC ) {
        jb = min( n - jc, DGEMM_NC );
        for ( pc = 0; pc < k; pc += DGEMM_KC ) {
            pb = min( k - pc, DGEMM_KC );
#ifdef _PARALLEL_
            #pragma omp parallel for num_threads( bl_ic_nt ) private( j )
#endif
            for ( j = 0; j < jb; j += DGEMM_NR ) {
                packB_add_stra_ab3( min( jb - j, DGEMM_NR ), pb, &b0[ pc + (jc+j)*ldb ], &b1[ pc + (jc+j)*ldb ], ldb, &packB[ j * pb ] );
            }
#ifdef _PARALLEL_
            #pragma omp parallel num_threads( bl_ic_nt ) private( ic, ib, i )
#endif
            {
                int tid = omp_get_thread_num();
                int my_start;
                int my_end;
                bl_get_range( m, DGEMM_MR, &my_start, &my_end );
                for ( ic = my_start; ic < my_end; ic += DGEMM_MC ) {
                    ib = min( my_end - ic, DGEMM_MC );
                    for ( i = 0; i < ib; i += DGEMM_MR ) {
                        packA_add_stra_ab3( min( ib - i, DGEMM_MR ), pb, &a0[ pc*lda + (ic+i) ], lda, &packA[ tid * DGEMM_MC * pb + i * pb ] );
                    }
                    bl_macro_kernel_stra_ab( ib, jb, pb, packA + tid * DGEMM_MC * pb, packB, &M[ jc * ldM + ic ], ldM );
                }
            }
        }
    }
    M_Add3( m, n, c0, c1, ldc, M, ldM, bl_ic_nt );
    free( M );
}

// M4 = (1 * a_0 + 1 * a_1) * (1 * b_3);  c_0 += -1 * M4;  c_1 += 1 * M4;
void bl_dgemm_straprim_ab4( int m, int n, int k, double* a0, double* a1, int lda, double* b0, int ldb, double* c0, double* c1, int ldc, double *packA, double *packB, int bl_ic_nt ) {
    int i, j, p, ic, ib, jc, jb, pc, pb;
    int ldM = m, nM = n;
    double *M = bl_malloc_aligned( ldM, nM, sizeof(double) );
    memset( M, 0, sizeof(double) * ldM * nM );
    for ( jc = 0; jc < n; jc += DGEMM_NC ) {
        jb = min( n - jc, DGEMM_NC );
        for ( pc = 0; pc < k; pc += DGEMM_KC ) {
            pb = min( k - pc, DGEMM_KC );
#ifdef _PARALLEL_
            #pragma omp parallel for num_threads( bl_ic_nt ) private( j )
#endif
            for ( j = 0; j < jb; j += DGEMM_NR ) {
                packB_add_stra_ab4( min( jb - j, DGEMM_NR ), pb, &b0[ pc + (jc+j)*ldb ], ldb, &packB[ j * pb ] );
            }
#ifdef _PARALLEL_
            #pragma omp parallel num_threads( bl_ic_nt ) private( ic, ib, i )
#endif
            {
                int tid = omp_get_thread_num();
                int my_start;
                int my_end;
                bl_get_range( m, DGEMM_MR, &my_start, &my_end );
                for ( ic = my_start; ic < my_end; ic += DGEMM_MC ) {
                    ib = min( my_end - ic, DGEMM_MC );
                    for ( i = 0; i < ib; i += DGEMM_MR ) {
                        packA_add_stra_ab4( min( ib - i, DGEMM_MR ), pb, &a0[ pc*lda + (ic+i) ], &a1[ pc*lda + (ic+i) ], lda, &packA[ tid * DGEMM_MC * pb + i * pb ] );
                    }
                    bl_macro_kernel_stra_ab( ib, jb, pb, packA + tid * DGEMM_MC * pb, packB, &M[ jc * ldM + ic ], ldM );
                }
            }
        }
    }
    M_Add4( m, n, c0, c1, ldc, M, ldM, bl_ic_nt );
    free( M );
}

// M5 = (-1 * a_0 + 1 * a_2) * (1 * b_0 + 1 * b_1);  c_3 += 1 * M5;
void bl_dgemm_straprim_ab5( int m, int n, int k, double* a0, double* a1, int lda, double* b0, double* b1, int ldb, double* c0, int ldc, double *packA, double *packB, int bl_ic_nt ) {
    int i, j, p, ic, ib, jc, jb, pc, pb;
    int ldM = m, nM = n;
    double *M = bl_malloc_aligned( ldM, nM, sizeof(double) );
    memset( M, 0, sizeof(double) * ldM * nM );
    for ( jc = 0; jc < n; jc += DGEMM_NC ) {
        jb = min( n - jc, DGEMM_NC );
        for ( pc = 0; pc < k; pc += DGEMM_KC ) {
            pb = min( k - pc, DGEMM_KC );
#ifdef _PARALLEL_
            #pragma omp parallel for num_threads( bl_ic_nt ) private( j )
#endif
            for ( j = 0; j < jb; j += DGEMM_NR ) {
                packB_add_stra_ab5( min( jb - j, DGEMM_NR ), pb, &b0[ pc + (jc+j)*ldb ], &b1[ pc + (jc+j)*ldb ], ldb, &packB[ j * pb ] );
            }
#ifdef _PARALLEL_
            #pragma omp parallel num_threads( bl_ic_nt ) private( ic, ib, i )
#endif
            {
                int tid = omp_get_thread_num();
                int my_start;
                int my_end;
                bl_get_range( m, DGEMM_MR, &my_start, &my_end );
                for ( ic = my_start; ic < my_end; ic += DGEMM_MC ) {
                    ib = min( my_end - ic, DGEMM_MC );
                    for ( i = 0; i < ib; i += DGEMM_MR ) {
                        packA_add_stra_ab5( min( ib - i, DGEMM_MR ), pb, &a0[ pc*lda + (ic+i) ], &a1[ pc*lda + (ic+i) ], lda, &packA[ tid * DGEMM_MC * pb + i * pb ] );
                    }
                    bl_macro_kernel_stra_ab( ib, jb, pb, packA + tid * DGEMM_MC * pb, packB, &M[ jc * ldM + ic ], ldM );
                }
            }
        }
    }
    M_Add5( m, n, c0, ldc, M, ldM, bl_ic_nt );
    free( M );
}

// M6 = (1 * a_1 + -1 * a_3) * (1 * b_2 + 1 * b_3);  c_0 += 1 * M6;
void bl_dgemm_straprim_ab6( int m, int n, int k, double* a0, double* a1, int lda, double* b0, double* b1, int ldb, double* c0, int ldc, double *packA, double *packB, int bl_ic_nt ) {
    int i, j, p, ic, ib, jc, jb, pc, pb;
    int ldM = m, nM = n;
    double *M = bl_malloc_aligned( ldM, nM, sizeof(double) );
    memset( M, 0, sizeof(double) * ldM * nM );
    for ( jc = 0; jc < n; jc += DGEMM_NC ) {
        jb = min( n - jc, DGEMM_NC );
        for ( pc = 0; pc < k; pc += DGEMM_KC ) {
            pb = min( k - pc, DGEMM_KC );
#ifdef _PARALLEL_
            #pragma omp parallel for num_threads( bl_ic_nt ) private( j )
#endif
            for ( j = 0; j < jb; j += DGEMM_NR ) {
                packB_add_stra_ab6( min( jb - j, DGEMM_NR ), pb, &b0[ pc + (jc+j)*ldb ], &b1[ pc + (jc+j)*ldb ], ldb, &packB[ j * pb ] );
            }
#ifdef _PARALLEL_
            #pragma omp parallel num_threads( bl_ic_nt ) private( ic, ib, i )
#endif
            {
                int tid = omp_get_thread_num();
                int my_start;
                int my_end;
                bl_get_range( m, DGEMM_MR, &my_start, &my_end );
                for ( ic = my_start; ic < my_end; ic += DGEMM_MC ) {
                    ib = min( my_end - ic, DGEMM_MC );
                    for ( i = 0; i < ib; i += DGEMM_MR ) {
                        packA_add_stra_ab6( min( ib - i, DGEMM_MR ), pb, &a0[ pc*lda + (ic+i) ], &a1[ pc*lda + (ic+i) ], lda, &packA[ tid * DGEMM_MC * pb + i * pb ] );
                    }
                    bl_macro_kernel_stra_ab( ib, jb, pb, packA + tid * DGEMM_MC * pb, packB, &M[ jc * ldM + ic ], ldM );
                }
            }
        }
    }
    M_Add6( m, n, c0, ldc, M, ldM, bl_ic_nt );
    free( M );
}

void bl_dgemm_strassen_ab( int m, int n, int k, double *XA, int lda, double *XB, int ldb, double *XC, int ldc )
{
    double *packA, *packB;
    char *str;
    int  bl_ic_nt;

    // Early return if possible
    if ( m == 0 || n == 0 || k == 0 ) {
        printf( "bl_dgemm_strassen_ab(): early return\n" );
        return;
    }

    // sequential is the default situation
    bl_ic_nt = 1;
    // check the environment variable
    str = getenv( "BLISLAB_IC_NT" );
    if ( str != NULL ) {
        bl_ic_nt = (int)strtol( str, NULL, 10 );
    }

    // Allocate packing buffers
    packA  = bl_malloc_aligned( DGEMM_KC, ( DGEMM_MC + 1 ) * bl_ic_nt, sizeof(double) );
    packB  = bl_malloc_aligned( DGEMM_KC, ( DGEMM_NC + 1 )           , sizeof(double) );

    int ms, ks, ns;
    int md, kd, nd;
    int mr, kr, nr;
    double *a = XA, *b= XB, *c = XC;

    mr = m % ( 2 * DGEMM_MR ), kr = k % ( 2 ), nr = n % ( 2 * DGEMM_NR );
    md = m - mr, kd = k - kr, nd = n - nr;


    ms=md, ks=kd, ns=nd;
    double *a_0, *a_1, *a_2, *a_3;
    bl_acquire_mpart( ms, ks, a, lda, 2, 2, 0, 0, &a_0 );
    bl_acquire_mpart( ms, ks, a, lda, 2, 2, 0, 1, &a_1 );
    bl_acquire_mpart( ms, ks, a, lda, 2, 2, 1, 0, &a_2 );
    bl_acquire_mpart( ms, ks, a, lda, 2, 2, 1, 1, &a_3 );


    ms=md, ks=kd, ns=nd;
    double *b_0, *b_1, *b_2, *b_3;
    bl_acquire_mpart( ks, ns, b, ldb, 2, 2, 0, 0, &b_0 );
    bl_acquire_mpart( ks, ns, b, ldb, 2, 2, 0, 1, &b_1 );
    bl_acquire_mpart( ks, ns, b, ldb, 2, 2, 1, 0, &b_2 );
    bl_acquire_mpart( ks, ns, b, ldb, 2, 2, 1, 1, &b_3 );


    ms=md, ks=kd, ns=nd;
    double *c_0, *c_1, *c_2, *c_3;
    bl_acquire_mpart( ms, ns, c, ldc, 2, 2, 0, 0, &c_0 );
    bl_acquire_mpart( ms, ns, c, ldc, 2, 2, 0, 1, &c_1 );
    bl_acquire_mpart( ms, ns, c, ldc, 2, 2, 1, 0, &c_2 );
    bl_acquire_mpart( ms, ns, c, ldc, 2, 2, 1, 1, &c_3 );


    ms=ms/2, ks=ks/2, ns=ns/2;


    // M0 = (1 * a_0 + 1 * a_3) * (1 * b_0 + 1 * b_3);  c_0 += 1 * M0;  c_3 += 1 * M0;
    bl_dgemm_straprim_ab0( ms, ns, ks, a_0, a_3, lda, b_0, b_3, ldb, c_0, c_3, ldc, packA, packB, bl_ic_nt );
    // M1 = (1 * a_2 + 1 * a_3) * (1 * b_0);  c_2 += 1 * M1;  c_3 += -1 * M1;
    bl_dgemm_straprim_ab1( ms, ns, ks, a_2, a_3, lda, b_0, ldb, c_2, c_3, ldc, packA, packB, bl_ic_nt );
    // M2 = (1 * a_0) * (1 * b_1 + -1 * b_3);  c_1 += 1 * M2;  c_3 += 1 * M2;
    bl_dgemm_straprim_ab2( ms, ns, ks, a_0, lda, b_1, b_3, ldb, c_1, c_3, ldc, packA, packB, bl_ic_nt );
    // M3 = (1 * a_3) * (-1 * b_0 + 1 * b_2);  c_0 += 1 * M3;  c_2 += 1 * M3;
    bl_dgemm_straprim_ab3( ms, ns, ks, a_3, lda, b_0, b_2, ldb, c_0, c_2, ldc, packA, packB, bl_ic_nt );
    // M4 = (1 * a_0 + 1 * a_1) * (1 * b_3);  c_0 += -1 * M4;  c_1 += 1 * M4;
    bl_dgemm_straprim_ab4( ms, ns, ks, a_0, a_1, lda, b_3, ldb, c_0, c_1, ldc, packA, packB, bl_ic_nt );
    // M5 = (-1 * a_0 + 1 * a_2) * (1 * b_0 + 1 * b_1);  c_3 += 1 * M5;
    bl_dgemm_straprim_ab5( ms, ns, ks, a_0, a_2, lda, b_0, b_1, ldb, c_3, ldc, packA, packB, bl_ic_nt );
    // M6 = (1 * a_1 + -1 * a_3) * (1 * b_2 + 1 * b_3);  c_0 += 1 * M6;
    bl_dgemm_straprim_ab6( ms, ns, ks, a_1, a_3, lda, b_2, b_3, ldb, c_0, ldc, packA, packB, bl_ic_nt );

    bl_dynamic_peeling( m, n, k, XA, lda, XB, ldb, XC, ldc, 2 * DGEMM_MR, 2, 2 * DGEMM_NR );

    free( packA );
    free( packB );
}
