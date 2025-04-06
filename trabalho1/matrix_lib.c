#include <stdio.h>
#include <stdlib.h>

#include "matrix_lib.h"

int scalar_matrix_mult(float scalar_value, matrix *m, matrix *r)
{
    // mais rapido
    float *pm = m->values;
    float *pr = r->values;

    for (int i = 0; i < m->rows * m->cols; i++)
    {
        *(pr++) = *(pm++) * scalar_value;
    }
    return 0;
}

/*
//mais ou menos
int scalar_matrix_mult(float scalar_value, matrix *m, matrix *r) {
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            r->values[i * r->cols + j] = m->values[i * m->cols + j] * scalar_value;
        }
    }
    return 0;
}
*/

/*
//mais lento
int scalar_matrix_mult(float scalar_value, matrix *m, matrix *r) {
    for (int j = 0; j < m->cols; j++) {
        for (int i = 0; i < m->rows; i++) {
            r->values[i * r->cols + j] = m->values[i * m->cols + j] * scalar_value;
        }
    }
    return 0;
}
*/

int matrix_matrix_mult(matrix *m1, matrix *m2, matrix *r)
{
    int rows = m1->rows;
    int cols = m2->cols;
    int common = m1->cols;

    // Inicializa a matriz resultado
    for (int i = 0; i < rows * cols; i++)
        r->values[i] = 0;

    // Multiplicação
    for (int i = 0; i < rows; i++)
    {
        for (int k = 0; k < common; k++)
        {
            float m1_val = m1->values[i * common + k];
            for (int j = 0; j < cols; j++)
            {
                r->values[i * cols + j] += m1_val * m2->values[k * cols + j];
            }
        }
    }

    return 0;
}

// ./gera-matrix matrix-test 8 8
// ./gera-matrix matrix-test2 8 8
// ./gera-matrix matrix-result 8 8
// ./gera-matrix matrix-result2 8 8
// ./mlt -s 10 -r 8 -c 8 -C 8 -m matrix-test -M matrix-test2 -o matrix-result -O matri-result2