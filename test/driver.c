#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <stdlib.h>
#include <ctype.h>

#include "blis.h"
#include "bli_fmm.h"


enum DriverFlag {
    NONE,
    REP_FLAG,
    VAR_FLAG,
    FMM_FLAG,
    RANDOM_FLAG,
    REINDEX_A_FLAG,
    REINDEX_B_FLAG,
};

void my_mm(obj_t* A, obj_t* B, obj_t* C, dim_t m, dim_t n, dim_t k) {

    void*  buf_A    = bli_obj_buffer_at_off( A ); 
    inc_t  rs_A     = bli_obj_row_stride( A ); 
    inc_t  cs_A     = bli_obj_col_stride( A ); 
    double *buf_Aptr = buf_A;

    void*  buf_B    = bli_obj_buffer_at_off( B ); 
    inc_t  rs_B     = bli_obj_row_stride( B ); 
    inc_t  cs_B    = bli_obj_col_stride( B ); 
    double *buf_Bptr = buf_B;

    void*  buf_C    = bli_obj_buffer_at_off( C ); 
    inc_t  rs_C     = bli_obj_row_stride( C ); 
    inc_t  cs_C    = bli_obj_col_stride( C ); 
    double *buf_Cptr = buf_C;

    for (dim_t i = 0; i < m; i++) {
        for (dim_t j = 0; j < n; j++) {
            for(dim_t p = 0; p < k; p++) {
                buf_Cptr[i*rs_C + j * cs_C] += buf_Aptr[i*rs_A + p*cs_A] * buf_Bptr[p*rs_B + j*cs_B];
            }
        }
    }
}

void run(dim_t m, dim_t n, dim_t k, fmm_t* fmm, int nreps)
{
    // fmm_t fmm_o = new_fmm("classical.txt");
    // fmm_t* fmm = &fmm_o;
    obj_t* null = 0;

    int failed = 0;

    num_t dt;
    inc_t rsC, csC;
    inc_t rsA, csA;
    inc_t rsB, csB;
    side_t side;

    obj_t A, B, C, C_ref, C_reset, diffM;
    obj_t* alpha;
    obj_t* beta;

    int    i, j;

    double tmp, error, flops;
    double ref_beg, bl_dgemm_beg, bl_dgemm_time;
    int    nrepeats;
    int    lda, ldb, ldc, ldc_ref;
    double ref_rectime, bl_dgemm_rectime, bl_dgemm_rectime_adjusted;
    double diff;

    dt = BLIS_DOUBLE;

    rsC = n; csC = 1;
    rsA = k; csA = 1;
    rsB = n; csB = 1;

    bli_obj_create( dt, m, n, 0, 0, &C );
    bli_obj_create( dt, m, n, 0, 0, &C_ref );
    bli_obj_create( dt, m, n, 0, 0, &C_reset );
    bli_obj_create( dt, m, n, 0, 0, &diffM );

    bli_obj_create( dt, m, k, 0, 0, &A );
    bli_obj_create( dt, k, n, 0, 0, &B );

    // Set the scalars to use.
    alpha = &BLIS_ONE;
    beta  = &BLIS_ONE;

    bli_randm( &B );
    bli_randm( &A );
    bli_randm( &C );
    bli_copym( &C, &C_ref );
    bli_copym( &C, &C_reset );

    double times_pre[] = { 0.0, 0.0, 0.0, 0.0 };
    double times_adjusted[] = { 0.0, 0.0, 0.0, 0.0 };

    const double CLOCK_CALL_TIME = .033 / 2107000.0;

    if (fmm == 0)
    {
        for ( i = 0; i < nreps; i ++ ) {
            bli_copym( &C_reset, &C );
            bl_dgemm_beg = bl_clock();
            {
                bli_gemm( alpha, &A, &B, beta, &C);
                // my_mm(&A, &B, &C, m, n, k);
            }
            bl_dgemm_time = bl_clock() - bl_dgemm_beg;

            if ( i == 0 ) {
                bl_dgemm_rectime = bl_dgemm_time;
            } else {
                bl_dgemm_rectime = bl_dgemm_time < bl_dgemm_rectime ? bl_dgemm_time : bl_dgemm_rectime;
            }
        }
    }
    else
    {   
        for ( i = 0; i < nreps; i ++ ) {
            TIMES[0] = 0.0;
            TIMES[1] = 0.0;
            TIMES[2] = 0.0;
            TIMES[3] = 0.0;
            CLOCK_CALLS[0] = 0;
            CLOCK_CALLS[1] = 0;
            CLOCK_CALLS[2] = 0;
            CLOCK_CALLS[3] = 0;
            bli_copym( &C_reset, &C );
            bl_dgemm_beg = bl_clock();
            {
                bli_fmm( alpha, &A, &B, beta, &C, fmm);
            }
            bl_dgemm_time = bl_clock() - bl_dgemm_beg;

            printf("CLOCK CALLS: %d\n", CLOCK_CALLS[1]);
            double total_clock_calls = (double)(CLOCK_CALLS[1] + CLOCK_CALLS[2] + CLOCK_CALLS[3]);
            double adjusted_time = bl_dgemm_time - total_clock_calls * CLOCK_CALL_TIME;

            if ( i == 0  || adjusted_time < bl_dgemm_rectime) {
                bl_dgemm_rectime = bl_dgemm_time;
                bl_dgemm_rectime_adjusted = adjusted_time;
                times_adjusted[0] = TIMES[0] - CLOCK_CALLS[0] * CLOCK_CALL_TIME;
                times_adjusted[1] = TIMES[1] - CLOCK_CALLS[1] * CLOCK_CALL_TIME;
                times_adjusted[2] = TIMES[2] - CLOCK_CALLS[2] * CLOCK_CALL_TIME;
                times_adjusted[3] = TIMES[3] - CLOCK_CALLS[3] * CLOCK_CALL_TIME;
                times_pre[0] = TIMES[0];
                times_pre[1] = TIMES[1];
                times_pre[2] = TIMES[2];
                times_pre[3] = TIMES[3];
            }
        }
    }

          // ref
    {
        ref_beg = bl_clock();
        {
            bli_gemm( alpha, &A, &B, beta, &C_ref);
            // my_mm(&A, &B, &C_ref, m, n, k);
        }
        ref_rectime = bl_clock() - ref_beg;
    }

    double        resid;
    obj_t  norm;
    double junk;

    bli_obj_scalar_init_detached( dt, &norm );

    bli_copym( &C_ref, &diffM );

    bli_subm( &C, &diffM );
    bli_normfm( &diffM, &norm );
    bli_getsc( &norm, &resid, &junk );

    // Compute overall floating point operations.
    flops = ( m * n / ( 1000.0 * 1000.0 * 1000.0 ) ) * ( 2 * k );

    printf( "%5d\t %5d\t %5d\t %5.2lf\t %5.2lf\t %5.2g\n",
                m, n, k, flops / bl_dgemm_rectime, flops / ref_rectime, resid );
    printf("\nTIMES PRE:\n====GFLOPS: %5.2g \tTotal %5.2g\n====ACC %5.2g UKR_TOTAL %5.2g PACKB %5.2g PACKA %5.2g\n\n", flops/bl_dgemm_rectime, bl_dgemm_rectime, times_pre[0], times_pre[1], times_pre[2], times_pre[3]);
    printf("\nADJUSTED:\n====GFLOPS: %5.2g \tTotal %5.2g\n====ACC %5.2g UKR_TOTAL %5.2g PACKB %5.2g PACKA %5.2g\n\n", flops/bl_dgemm_rectime_adjusted, bl_dgemm_rectime_adjusted, times_adjusted[0], times_adjusted[1], times_adjusted[2], times_adjusted[3]);

    fflush(stdout);

    // Free the objects.
    bli_obj_free( &A );
    bli_obj_free( &B );
    bli_obj_free( &C );
    bli_obj_free( &C_ref );
    bli_obj_free( &C_reset );
    bli_obj_free( &diffM );

    return failed;
}

bool is_numeric(const char *str) 
{
    while(*str != '\0')
    {
        if(*str < '0' || *str > '9')
            return false;
        str++;
    }
    return true;
}

int main( int argc, char *argv[] )
{

    // allow for selection of algorithm and nesting level
    // allow for mixing algorithms

    // allow for arbitrary size

    // allow for different variants

    // allow for running baseline / reference

    // driver.x m n k --var -2 (ref) --reps 3

    assert(argc >= 4);

    int m = atoi(argv[1]);
    int n = atoi(argv[2]);
    int k = atoi(argv[3]);

    // if (1) {
    //     fmm_t final_fmm = new_fmm("classical.txt");
    //     run(m, n, k, 1, -1, &final_fmm);
    //     free_fmm(&final_fmm);
    //     return 0;
    // }

    int nreps = 3;
    int variant = -1;
    bool reindex_a = false;
    bool reindex_b = false;
    bool randomize = false;

    enum DriverFlag current_flag = NONE;

    fmm_t final_fmm = new_fmm("classical.txt");
    int num_layers = 1;

    for (int i = 4; i < argc; i++) {

        char* arg = argv[i];

        switch (current_flag) {
            case NONE:
                if (arg[0] == '-') {
                    switch (arg[1]) {
                        case 'a':
                            current_flag = REINDEX_A_FLAG;
                        break;
                        case 'b':
                            current_flag = REINDEX_B_FLAG;
                        break;
                        case 'r':
                            current_flag = REP_FLAG;
                        break;
                        case 'v':
                            current_flag = VAR_FLAG;
                        break;
                        case 'z':
                            current_flag = RANDOM_FLAG;
                        case 'f':
                            ++i;

                            // cleanup
                            if (num_layers != 0) {
                                free_fmm(&final_fmm);
                            }

                            if (!is_numeric(argv[i])) {
                                printf("%s\n\n", argv[i]);
                                printf("Incorrect format of driver parameters\n");
                                bli_abort();
                            }
                            num_layers = atoi(argv[i]);
                            ++i;
                            assert(num_layers + i <= argc);
                            printf("Reading FMM recipe:\n");
                            for (int j = 0; j < num_layers; j++) {
                                printf("%d -> %s\n", j, argv[i]);
                                fmm_t child_fmm = new_fmm(argv[i]);
                                if (j == 0) {
                                    final_fmm = child_fmm;
                                    i++;
                                    continue;
                                }
                                final_fmm = nest_fmm(&final_fmm, &child_fmm);
                                ++i;
                            }

                        break;
                    }
                }
                continue;
            break;
            case REINDEX_A_FLAG:
                reindex_a = (arg[0] == '1');
            break;
            case REINDEX_B_FLAG:
                reindex_b = (arg[0] == '1');
            break;
            case REP_FLAG:
                nreps = atoi(arg);
            break;
            case VAR_FLAG:
                variant = atoi(arg);
            break;
            case RANDOM_FLAG:
                randomize = true;
            break;
        }
        current_flag = NONE;
    }

    if (variant < -2 || variant > 2 || nreps < 1) {
        printf("Illegal arguments to driver...\n");
        bli_abort();
        return 1;
    }  

    if (randomize) {
        fmm_shuffle_columns(&final_fmm);
    }

    final_fmm.reindex_a = reindex_a;
    final_fmm.reindex_b = reindex_b;
    final_fmm.variant = variant;

    if (variant == -2)
    {
        run(m, n, k, 0, nreps);
    }
    else {
        printf("m %d\tn %d\tk %d\n", m, n, k);
        printf("nreps %d \t variant %d reindex_a %d reindex_b %d\n\n",
            nreps, variant, reindex_a, reindex_b);
        printf("fmm->R %d mt %d nt %d kt %d\n",
            final_fmm.R, final_fmm.m_tilde, final_fmm.n_tilde, final_fmm.k_tilde);

        run(m, n, k, &final_fmm, nreps);
        printf("\n\n");
    }

    free_fmm(&final_fmm);

    return 0;
}
