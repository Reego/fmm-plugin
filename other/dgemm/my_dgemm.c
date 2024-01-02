#include "bl_dgemm_kernel.h"
#include "bl_dgemm.h"

static inline void packA_mcxkc_d(
        int    m,
        int    k,
        double *XA,
        int    ldXA,
        int    offseta,
        double *packA
        )
{
    int    i, p;
    double *a_pntr[ DGEMM_MR ];

#if 1
    if ( m == DGEMM_MR ) {
    /* Full row size micro-panel.*/
        for ( int p=0; p<k; p++ ) {
            for ( int i=0; i<DGEMM_MR; i++ ) {
                *packA ++ = *(XA + (offseta + i) + p *ldXA);
            }
        }
    }
    else {
        for ( int p=0; p<k; p++ ) {
            for ( int i=0; i<m; i++ ) {
                *packA ++ = *(XA + (offseta + i) + p *ldXA);
            }
            for ( int i=m; i<DGEMM_MR; i++ ) {
                *packA++ = 0.0;
            }
        }
    }
#else

    for ( i = 0; i < m; i ++ ) {
        a_pntr[ i ] = XA + ( offseta + i );
    }

    for ( i = m; i < DGEMM_MR; i ++ ) {
        a_pntr[ i ] = XA + ( offseta + 0 );
    }

    for ( p = 0; p < k; p ++ ) {
        for ( i = 0; i < DGEMM_MR; i ++ ) {
            *packA = *a_pntr[ i ];
            packA ++;
            a_pntr[ i ] = a_pntr[ i ] + ldXA;
        }
    }
#endif
}


/*
 * --------------------------------------------------------------------------
 */

static inline void packB_kcxnc_d(
        int    n,
        int    k,
        double *XB,
        int    ldXB, // ldXB is the original k
        int    offsetb,
        double *packB
        )
{
    int    j, p; 
    double *b_pntr[ DGEMM_NR ];

#if 1
    if ( n == DGEMM_NR ) {
    /* Full column width micro-panel.*/
        for ( int p=0; p<k; p++ )
        {
            for ( int j=0; j<DGEMM_NR; j++ )
            {
                *packB++ = *(XB + (ldXB * (offsetb + j)) + p);
            }
        }
    }
    else {
        for ( int p=0; p<k; p++ )
        {
            for ( int j=0; j<n; j++ )
            {
                *packB++ = *(XB + (ldXB * (offsetb + j)) + p);
            }
            for ( int j=n; j < DGEMM_NR; j++ )
            {
                *packB++ = 0.0;
            }
        }
    }

#else
    for ( j = 0; j < n; j ++ ) {
        b_pntr[ j ] = XB + ldXB * ( offsetb + j );
    }

    for ( j = n; j < DGEMM_NR; j ++ ) {
        b_pntr[ j ] = XB + ldXB * ( offsetb + 0 );
    }

    for ( p = 0; p < k; p ++ ) {
        for ( j = 0; j < DGEMM_NR; j ++ ) {
            *packB ++ = *b_pntr[ j ] ++;
        }
    }
#endif
}

/*
 * --------------------------------------------------------------------------
 */
void bl_macro_kernel(
        int    m,
        int    n,
        int    k,
        double *packA,
        double *packB,
        double *C,
        int    ldc
        )
{
    int bl_ic_nt;
    int    i, ii, j;
    aux_t  aux;
    char *str;

    aux.b_next = packB;

    // We can also parallelize with OMP here.
    //// sequential is the default situation
    //bl_ic_nt = 1;
    //// check the environment variable
    //str = getenv( "BLISLAB_IC_NT" );
    //if ( str != NULL ) {
    //    bl_ic_nt = (int)strtol( str, NULL, 10 );
    //}
    //#pragma omp parallel for num_threads( bl_ic_nt ) private( j, i, aux )
    for ( j = 0; j < n; j += DGEMM_NR ) {                        // 2-th loop around micro-kernel
        aux.n  = min( n - j, DGEMM_NR );
        for ( i = 0; i < m; i += DGEMM_MR ) {                    // 1-th loop around micro-kernel
            aux.m = min( m - i, DGEMM_MR );
            if ( i + DGEMM_MR >= m ) {
                aux.b_next += DGEMM_NR * k;
            }

#if 0
            __m256d gamma_0123_0 = _mm256_load_pd( &C[ j * ldc + i ] );
            __m256d gamma_0123_1 = _mm256_load_pd( &C[ (j+1) * ldc + i ] );
            __m256d gamma_0123_2 = _mm256_load_pd( &C[ (j+2) * ldc + i ] );
            __m256d gamma_0123_3 = _mm256_load_pd( &C[ (j+3) * ldc + i ] );
            __m256d gamma_0123_4 = _mm256_load_pd( &C[ (j+4) * ldc + i ] );
            __m256d gamma_0123_5 = _mm256_load_pd( &C[ (j+5) * ldc + i ] );

            __m256d gamma_4567_0 = _mm256_load_pd( &C[ j * ldc + i + 4 ] );
            __m256d gamma_4567_1 = _mm256_load_pd( &C[ (j+1) * ldc + i + 4 ] );
            __m256d gamma_4567_2 = _mm256_load_pd( &C[ (j+2) * ldc + i + 4] );
            __m256d gamma_4567_3 = _mm256_load_pd( &C[ (j+3) * ldc + i + 4] );
            __m256d gamma_4567_4 = _mm256_load_pd( &C[ (j+4) * ldc + i + 4] );
            __m256d gamma_4567_5 = _mm256_load_pd( &C[ (j+5) * ldc + i + 4] );

            double *aptr = &packA[ i * k ];
            double *bptr = &packB[ j * k ];

            for (int p = 0; p < k; p++ )
            {
                /* Declare vector register for load/broadcasting beta( p,j ) */
                __m256d beta_p_j;
    
                /* Declare vector registersx to hold the current column of A 
                and load them with the eight elements of that column. */
                __m256d alpha_0123_p = _mm256_loadu_pd( aptr );
                __m256d alpha_4567_p = _mm256_loadu_pd( aptr + 4 );

                /* Load/broadcast beta( p,0 ). */
                beta_p_j = _mm256_broadcast_sd( bptr );
                
                /* update the first column of C with the current column of A times
                beta ( p,0 ) */
                gamma_0123_0 = _mm256_fmadd_pd( alpha_0123_p, beta_p_j, gamma_0123_0 );
                gamma_4567_0 = _mm256_fmadd_pd( alpha_4567_p, beta_p_j, gamma_4567_0 );
                

                /* Load/broadcast beta( p,1 ). */
                beta_p_j = _mm256_broadcast_sd( bptr + 1 );
                
                /* update the second column of C with the current column of A times
                beta ( p,1 ) */
                gamma_0123_1 = _mm256_fmadd_pd( alpha_0123_p, beta_p_j, gamma_0123_1 );
                gamma_4567_1 = _mm256_fmadd_pd( alpha_4567_p, beta_p_j, gamma_4567_1 );

                /* Load/broadcast beta( p,2 ). */
                beta_p_j = _mm256_broadcast_sd( bptr + 2 );
                
                /* update the third column of C with the current column of A times
                beta ( p,2 ) */
                gamma_0123_2 = _mm256_fmadd_pd( alpha_0123_p, beta_p_j, gamma_0123_2 );
                gamma_4567_2 = _mm256_fmadd_pd( alpha_4567_p, beta_p_j, gamma_4567_2 );

                /* Load/broadcast beta( p,3 ). */
                beta_p_j = _mm256_broadcast_sd( bptr + 3 );
                
                /* update the fourth column of C with the current column of A times
                beta ( p,3 ) */
                gamma_0123_3 = _mm256_fmadd_pd( alpha_0123_p, beta_p_j, gamma_0123_3 );
                gamma_4567_3 = _mm256_fmadd_pd( alpha_4567_p, beta_p_j, gamma_4567_3 );

                /* Load/broadcast beta( p,4 ). */
                beta_p_j = _mm256_broadcast_sd( bptr + 4 );
                
                /* update the fifth column of C with the current column of A times
                beta ( p,4 ) */
                gamma_0123_4 = _mm256_fmadd_pd( alpha_0123_p, beta_p_j, gamma_0123_4 );
                gamma_4567_4 = _mm256_fmadd_pd( alpha_4567_p, beta_p_j, gamma_4567_4 );

                /* Load/broadcast beta( p,5 ). */
                beta_p_j = _mm256_broadcast_sd( bptr + 5 );
                
                /* update the sixth column of C with the current column of A times
                beta ( p,5 ) */
                gamma_0123_5 = _mm256_fmadd_pd( alpha_0123_p, beta_p_j, gamma_0123_5 );
                gamma_4567_5 = _mm256_fmadd_pd( alpha_4567_p, beta_p_j, gamma_4567_5 );

                aptr += DGEMM_MR;
                bptr += DGEMM_NR;
            }

            _mm256_store_pd( &C[ (j + 0 ) * ldc + i ], gamma_0123_0 );
            _mm256_store_pd( &C[ (j + 1 ) * ldc + i ], gamma_0123_1 );
            _mm256_store_pd( &C[ (j + 2 ) * ldc + i ], gamma_0123_2 );
            _mm256_store_pd( &C[ (j + 3 ) * ldc + i ], gamma_0123_3 );
            _mm256_store_pd( &C[ (j + 4 ) * ldc + i ], gamma_0123_4 );
            _mm256_store_pd( &C[ (j + 5 ) * ldc + i ], gamma_0123_5 );
            _mm256_store_pd( &C[ (j + 0 ) * ldc + i + 4 ], gamma_4567_0 );
            _mm256_store_pd( &C[ (j + 1 ) * ldc + i + 4 ], gamma_4567_1 );
            _mm256_store_pd( &C[ (j + 2 ) * ldc + i + 4 ], gamma_4567_2 );
            _mm256_store_pd( &C[ (j + 3 ) * ldc + i + 4 ], gamma_4567_3 );
            _mm256_store_pd( &C[ (j + 4 ) * ldc + i + 4 ], gamma_4567_4 );
            _mm256_store_pd( &C[ (j + 5 ) * ldc + i + 4 ], gamma_4567_5 );

#else 
            ( *bl_micro_kernel ) (
                    k,
                    &packA[ i * k ],
                    &packB[ j * k ],
                    &C[ j * ldc + i ],
                    (unsigned long long) ldc,
                    //&aux
                    NULL
                    );
#endif
        }                                                        // 1-th loop around micro-kernel
    }                                                            // 2-th loop around micro-kernel
}

// C must be aligned
void bl_dgemm(
        int    m,
        int    n,
        int    k,
        double *XA,
        int    lda,
        double *XB,
        int    ldb,
        double *C,        // must be aligned
        int    ldc        // ldc must also be aligned
        )
{
    int    i, j, p, bl_ic_nt;
    int    ic, ib, jc, jb, pc, pb;
    int    ir, jr;
    double *packA, *packB;
    char   *str;

    // Early return if possible
    if ( m == 0 || n == 0 || k == 0 ) {
        printf( "bl_dgemm(): early return\n" );
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

    for ( jc = 0; jc < n; jc += DGEMM_NC ) {                                       // 5-th loop around micro-kernel
        jb = min( n - jc, DGEMM_NC );
        for ( pc = 0; pc < k; pc += DGEMM_KC ) {                                   // 4-th loop around micro-kernel
            pb = min( k - pc, DGEMM_KC );

            #pragma omp parallel for num_threads( bl_ic_nt ) private( jr )
            for ( j = 0; j < jb; j += DGEMM_NR ) {
                packB_kcxnc_d(
                        min( jb - j, DGEMM_NR ),
                        pb,
                        &XB[ pc ],
                        ldb,
                        jc + j,
                        &packB[ j * pb ]
                        );
            }

            //#pragma omp parallel for num_threads( bl_ic_nt ) private( ic, ib, i, ir )
            #pragma omp parallel num_threads( bl_ic_nt ) private( ic, ib, i, ir )
            {
                int     tid      = omp_get_thread_num();
                int     my_start;
                int     my_end;

                bl_get_range( m, DGEMM_MR, &my_start, &my_end );

                for ( ic = my_start; ic < my_end; ic += DGEMM_MC ) {              // 3-rd loop around micro-kernel

                    ib = min( my_end - ic, DGEMM_MC );

                    for ( i = 0; i < ib; i += DGEMM_MR ) {
                        packA_mcxkc_d(
                                min( ib - i, DGEMM_MR ),
                                pb,
                                &XA[ pc * lda ],
                                lda,
                                ic + i,
                                &packA[ tid * DGEMM_MC * pb + i * pb ]
                                );
                    }

                    bl_macro_kernel(
                            ib,
                            jb,
                            pb,
                            packA  + tid * DGEMM_MC * pb,
                            packB,
                            &C[ jc * ldc + ic ], 
                            ldc
                            );

                }                                                                // End 3.rd loop around micro-kernel

            }
        }                                                                        // End 4.th loop around micro-kernel
    }                                                                            // End 5.th loop around micro-kernel
    free( packB );

    free( packA );
    
}


