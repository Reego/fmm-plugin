#include "blis.h"
#include "bli_fmm.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define LINE_N 256

#define _U( i,j ) fmm.U[ (i)*fmm.R + (j) ]
#define _V( i,j ) fmm.V[ (i)*fmm.R + (j) ]
#define _W( i,j ) fmm.W[ (i)*fmm.R + (j) ]

#define __U( i,j ) fmm->U[ (i)*fmm->R + (j) ]
#define __V( i,j ) fmm->V[ (i)*fmm->R + (j) ]
#define __W( i,j ) fmm->W[ (i)*fmm->R + (j) ]

int _aparts(fmm_t* fmm) {
    return fmm->m_tilde * fmm->k_tilde;
}

int _bparts(fmm_t* fmm) {
    return fmm->k_tilde * fmm->n_tilde;
}

int _cparts(fmm_t* fmm) {
    return fmm->m_tilde * fmm->n_tilde;
}

void fmm_rearrange(
    int* U,
    int R,
    int m_tilde_out,
    int k_tilde_out,
    int m_tilde_in,
    int k_tilde_in
)
{
    int m_tilde = m_tilde_out * m_tilde_in;
    int k_tilde = k_tilde_out * k_tilde_in;

    int* dest = (int*) malloc(sizeof(int) * m_tilde * k_tilde);

    for (int col = 0; col < R; col++)
    {
        int* column = &U[col];

        for (int i = 0; i < m_tilde * k_tilde; i++) {

            int block_index = i / (m_tilde_in * k_tilde_in);
            int block_level_row = block_index / k_tilde_out;
            int block_level_col = block_index % k_tilde_out;

            int i_left = i % (m_tilde * k_tilde);

            int inner_index = i % (m_tilde_in * k_tilde_in);
            int row = block_level_row * k_tilde_in + inner_index / k_tilde_in;
            int col = block_level_col * m_tilde_in + inner_index % k_tilde_in;

            int dest_index = row * k_tilde_out * k_tilde_in + col;

            dest[dest_index] = *(column + R * i);
        }

        for (int row = 0; row < m_tilde * k_tilde; row++) {
            *(column + R * row) = dest[row];
        }
    }

    free(dest);
}

void shuffle(int *array, size_t n)
{
    if (n > 1) 
    {
        size_t i;
        for (i = 0; i < n - 1; i++) 
        {
          size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
          int t = array[j];
          array[j] = array[i];
          array[i] = t;
        }
    }
}

void reshuffle_columns(int* order, float* U, int m, int k, int R) {

    float* buffer = (float*) malloc(sizeof(float) * m * k * R);

    memcpy(U, buffer, sizeof(float) * m * k * R);

    for (int r = 0; r < R; r++)
    {
        for (int i = 0; i < m * k; i++)
        {
            U[i * R + r] = U[i * R + order[r]];
        }
    }

    free(buffer);
}

void fmm_shuffle_columns_ex(fmm_t* fmm, int* order)
{
    shuffle(order, fmm->R);

    reshuffle_columns(order, fmm->U, fmm->m_tilde, fmm->k_tilde, fmm->R);
    reshuffle_columns(order, fmm->V, fmm->k_tilde, fmm->n_tilde, fmm->R);
    reshuffle_columns(order, fmm->W, fmm->m_tilde, fmm->n_tilde, fmm->R);
}

void fmm_shuffle_columns(fmm_t* fmm)
{
    int* order = (int*) malloc(sizeof(int) * fmm->R);

    fmm_shuffle_columns_ex(fmm, order);

    free(order);
}

void nest_fmm_helper(
    fmm_t* fmm_a,
    fmm_t* fmm_b,
    fmm_t* dest,
    int* buffer_a,
    int* buffer_b,
    int* buffer_dest,
    int a_parts,
    int b_parts
)
{
    for (int row_A_U = 0; row_A_U < a_parts; row_A_U++) {
        for (int col_A_U = 0; col_A_U < fmm_a->R; col_A_U++) {
            for (int row_B_U = 0; row_B_U < b_parts; row_B_U++) {
                for (int col_B_U = 0; col_B_U < fmm_b->R; col_B_U++) {

                    float a_element = buffer_a[row_A_U * fmm_a->R + col_A_U];
                    float b_element = buffer_b[row_B_U * fmm_b->R + col_B_U];

                    int dest_row = row_A_U * b_parts + row_B_U;
                    int dest_col = col_A_U * fmm_b->R + col_B_U;

                    int index = dest_row * dest->R + dest_col;
                    buffer_dest[index] = a_element * b_element;
                }
            }
        }
    }
}

fmm_t nest_fmm(fmm_t* fmm_a, fmm_t* fmm_b) {

    fmm_t fmm;

    fmm.m_tilde = fmm_a->m_tilde * fmm_b->m_tilde;
    fmm.k_tilde = fmm_a->k_tilde * fmm_b->k_tilde;
    fmm.n_tilde = fmm_a->n_tilde * fmm_b->n_tilde;

    fmm.R = fmm_a->R * fmm_b->R;

    int a_aparts = _aparts(fmm_a);
    int a_bparts = _bparts(fmm_a);
    int a_cparts = _cparts(fmm_a);

    int b_aparts = _aparts(fmm_b);
    int b_U_size = b_aparts * fmm_b->R;

    int b_bparts = _bparts(fmm_b);
    int b_V_size = b_bparts * fmm_b->R;

    int b_cparts = _cparts(fmm_b);
    int b_W_size = b_cparts * fmm_b->R;

    int aparts = _aparts(&fmm);
    int bparts = _bparts(&fmm);
    int cparts = _cparts(&fmm);

    fmm.U = (int*) malloc( sizeof(int) * fmm.R * aparts);
    fmm.V = (int*) malloc( sizeof(int) * fmm.R * bparts);
    fmm.W = (int*) malloc( sizeof(int) * fmm.R * cparts);

    nest_fmm_helper(
        fmm_a,
        fmm_b,
        &fmm,
        fmm_a->U,
        fmm_b->U,
        fmm.U,
        a_aparts,
        b_aparts
    );

    nest_fmm_helper(
        fmm_a,
        fmm_b,
        &fmm,
        fmm_a->V,
        fmm_b->V,
        fmm.V,
        a_bparts,
        b_bparts
    );

    nest_fmm_helper(
        fmm_a,
        fmm_b,
        &fmm,
        fmm_a->W,
        fmm_b->W,
        fmm.W,
        a_cparts,
        b_cparts
    );

    fmm_rearrange(fmm.U, fmm.R, fmm_a->m_tilde, fmm_a->k_tilde, fmm_b->m_tilde, fmm_b->k_tilde);
    fmm_rearrange(fmm.V, fmm.R, fmm_a->k_tilde, fmm_a->n_tilde, fmm_b->k_tilde, fmm_b->n_tilde);
    fmm_rearrange(fmm.W, fmm.R, fmm_a->m_tilde, fmm_a->n_tilde, fmm_b->m_tilde, fmm_b->n_tilde);

    fmm.reindex_a = fmm_a->reindex_a;
    fmm.reindex_b = fmm_a->reindex_b;
    fmm.variant = fmm_a->variant;

    free_fmm(fmm_a);
    free_fmm(fmm_b);

    return fmm;
}

fmm_t new_fmm_ex(const char* file_name, int nest_level, int variant, bool reindex_a, bool reindex_b) {

    if (nest_level <= 1) {

        fmm_t fmm;

        FILE* fp = fopen(file_name, "r");
        char line[LINE_N];

        fgets(line, LINE_N, fp);

        sscanf(line, "%d %d %d %d", &fmm.m_tilde, &fmm.k_tilde, &fmm.n_tilde, &fmm.R);

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

        fmm.reindex_a = reindex_a;
        fmm.reindex_b = reindex_b;
        fmm.variant = variant;

        return fmm;
    }

    fmm_t fmm_a = new_fmm_ex(file_name, nest_level - 1, variant, reindex_a, reindex_b);
    fmm_t fmm_b = new_fmm_ex(file_name, 1, variant, reindex_a, reindex_b);
    fmm_t final_fmm = nest_fmm(&fmm_a, &fmm_b);

    return final_fmm;
}

fmm_t new_fmm(const char* file_name) {
    return new_fmm_ex(file_name, 1, -1, false, false);
}

void print_fmm(fmm_t* fmm) {

    printf("%d %d %d\t%d\n#\n", fmm->m_tilde, fmm->k_tilde, fmm->n_tilde, fmm->R);

    for (int a = 0; a < _aparts(fmm); a++) {
        for (int r = 0; r < fmm->R; r++) {
            printf("%d ", __U(a, r));
        }
        printf("\n");
    }
    printf("#\n");
    for (int i = 0; i < _bparts(fmm); i++) {
        for (int r = 0; r < fmm->R; r++) {
            printf("%d ", __V(i, r));
        }
        printf("\n");
    }
    printf("#\n");
    for (int i = 0; i < _cparts(fmm); i++) {
        for (int r = 0; r < fmm->R; r++) {
            printf("%d ", __W(i, r));
        }
        printf("\n");
    }
}

void free_fmm(fmm_t* fmm) {
    free(fmm->U);
    free(fmm->V);
    free(fmm->W);
}