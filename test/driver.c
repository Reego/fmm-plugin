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
    FMM_FLAG
};

void run(dim_t m, dim_t n, dim_t k, int nreps, int variant, fmm_t* fmm)
{
    obj_t* null = 0;

    int failed = 0;

    num_t dt;
    inc_t rsC, csC;
    inc_t rsA, csA;
    inc_t rsB, csB;
    side_t side;

    obj_t A, B, C, C_ref, diffM;
    obj_t* alpha;
    obj_t* beta;

    int    i, j;

    double tmp, error, flops;
    double ref_beg, bl_dgemm_beg, bl_dgemm_time;
    int    nrepeats;
    int    lda, ldb, ldc, ldc_ref;
    double ref_rectime, bl_dgemm_rectime;
    double diff;

    dt = BLIS_DOUBLE;

    rsC = n; csC = 1;
    rsA = k; csA = 1;
    rsB = n; csB = 1;

    bli_obj_create( dt, m, n, 0, 0, &C );
    bli_obj_create( dt, m, n, 0, 0, &C_ref );
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

    for ( i = 0; i < nreps; i ++ ) {
        bl_dgemm_beg = bl_clock();
        {
            bli_strassen_ab_ex_ex( alpha, &A, &B, beta, &C, fmm, variant);
            // bli_strassen_ab( alpha, &A, &B, beta, &C);
        }
        bl_dgemm_time = bl_clock() - bl_dgemm_beg;

        if ( i == 0 ) {
            bl_dgemm_rectime = bl_dgemm_time;
        } else {
            bl_dgemm_rectime = bl_dgemm_time < bl_dgemm_rectime ? bl_dgemm_time : bl_dgemm_rectime;
        }
    }

    // ref
    {
        ref_beg = bl_clock();
        {
            bli_gemm( alpha, &A, &B, beta, &C_ref);
        }
        ref_rectime = bl_clock() - ref_beg;
    }

    double        resid, resid_other;
    obj_t  norm;
    double junk;

    bli_obj_scalar_init_detached( dt, &norm );

    bli_copym( &C_ref, &diffM );

    bli_subm( &C, &diffM );
    bli_normfm( &diffM, &norm );
    bli_getsc( &norm, &resid, &junk );

    // Compute overall floating point operations.
    flops = ( m * n / ( 1000.0 * 1000.0 * 1000.0 ) ) * ( 2 * k );

    printf( "%5d\t %5d\t %5d\t %5.2lf\t %5.2lf\t %5.2g\t %5.2g\n",
                m, n, k, flops / bl_dgemm_rectime, flops / ref_rectime, resid, resid_other );

    fflush(stdout);

    // Free the objects.
    bli_obj_free( &A );
    bli_obj_free( &B );
    bli_obj_free( &C );
    bli_obj_free( &C_ref );
    bli_obj_free( &diffM );

    return failed;
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

    int nreps = 3;
    int variant = -2;

    enum DriverFlag current_flag = NONE;

    fmm_t* fmm_layers = (fmm_t*) malloc(sizeof(fmm_t));
    fmm_layers[0] = new_fmm("classical.txt");
    int num_layers = 1;

    for (int i = 4; i < argc; i++) {

        char* arg = argv[i];

        switch (current_flag) {
            case NONE:
                if (arg[0] == '-') {
                    switch (arg[1]) {
                        case 'r':
                            current_flag = REP_FLAG;
                        break;
                        case 'v':
                            current_flag = VAR_FLAG;
                        break;
                        case 'f':
                            ++i;

                            // cleanup
                            if (num_layers != 0) {
                                for (int j = 0; j < num_layers; j++) {
                                    free_fmm(&fmm_layers[j]);
                                }
                                free(fmm_layers);
                            }

                            if (!isnumber(argv[i])) {
                                printf("Incorrect format of driver parameters\n");
                                bli_abort();
                            }
                            num_layers = atoi(argv[i]);
                            fmm_layers = (fmm_t*) malloc(sizeof(fmm_t) * num_layers);
                            ++i;
                            assert(num_layers + i <= argc);
                            printf("Reading FMM recipe:\n");
                            for (int j = 0; j < num_layers; j++) {
                                printf("%d -> %s\n", argv[i]);
                                fmm_layers[j] = new_fmm(argv[i]);
                                ++i;
                            }

                        break;
                    }
                }
                continue;
            break;
            case REP_FLAG:
                nreps = atoi(arg);
            break;
            case VAR_FLAG:
                variant = atoi(arg);
            break;
        }
        current_flag = NONE;
    }

    if (variant < -2 || variant > 2 || nreps < 1) {
        printf("Illegal arguments to driver...\n");
        bli_abort();
        return 1;
    }

    printf("m %d\tn %d\tk %d\n", m, n, k);
    printf("nreps %d \t variant %d\n\n", nreps, variant);

    fmm_t final_fmm = fmm_layers[0];
    for (int j = 1; j < num_layers; j++) {
        fmm_t* temp = &final_fmm;
        final_fmm = nest_fmm(&final_fmm, &fmm_layers[j]);
        if (j != 1) {
            free_fmm(temp);
            free_fmm(&fmm_layers[j]);
        }
    }
    free(fmm_layers);

    run(m, n, k, nreps, variant, &final_fmm);

    free_fmm(&final_fmm);

    return 0;
}