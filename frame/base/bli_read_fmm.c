#include "bli_fmm.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define LINE_N 128

#define _U( i,j ) fmm.U[ (i)*fmm.R + (j) ]
#define _V( i,j ) fmm.V[ (i)*fmm.R + (j) ]
#define _W( i,j ) fmm.W[ (i)*fmm.R + (j) ]

void do_fmm_test() {
    fmm_t fmm = STRASSEN_FMM;

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 7; j++) {
            printf("%d ", _U(i, j));
        }
        printf("\n");
    }
}

fmm_t new_fmm(const char* file_name) {

    fmm_t fmm;

    FILE* fp = fopen(file_name, "r");
    char line[LINE_N];

    fgets(line, LINE_N, fp);

    sscanf(line, "%d %d %d %d", &fmm.m_tilde, &fmm.n_tilde, &fmm.k_tilde, &fmm.R);

    int aparts = fmm.m_tilde * fmm.k_tilde;
    int bparts = fmm.k_tilde * fmm.n_tilde;
    int cparts = fmm.m_tilde * fmm.n_tilde;

    fmm.U = (int*) malloc( sizeof(int) * fmm.R * aparts);
    fmm.V = (int*) malloc( sizeof(int) * fmm.R * bparts);
    fmm.W = (int*) malloc( sizeof(int) * fmm.R * cparts);

    int num_lines = 0;
    int offset = 0;
    while (fgets(line, LINE_N, fp) != NULL) {

        if (line[0] != 0 && line[0] == '#') {
            continue;
        }

        int* coefs;

        if (num_lines < aparts) {
            offset = num_lines;
            coefs = fmm.U;
        }
        else if (num_lines < aparts + bparts) {
            offset = num_lines - aparts;
            coefs = fmm.V;
        }
        else {
            offset = num_lines - aparts - bparts;
            coefs = fmm.W;
        }

        FILE* stream = fmemopen (line, strlen (line), "r");
        int num;
        int i = 0;

        while (fscanf (stream, "%d", &num) == 1) {
            *(coefs + fmm.R * offset + i) = num;
            ++i;
        }
        ++num_lines;
    }

    fclose(fp);

    return fmm;
}

void free_fmm(fmm_t fmm) {
    free(fmm.U);
    free(fmm.V);
    free(fmm.W);
}