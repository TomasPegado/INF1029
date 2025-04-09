#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>  // AVX intrinsics

#include "matrix_lib.h"

// int scalar_matrix_mult(float scalar_value, matrix *m, matrix *r)
// {
//     // mais rapido
//     float *pm = m->values;
//     float *pr = r->values;

//     for (int i = 0; i < m->rows * m->cols; i++)
//     {
//         *(pr++) = *(pm++) * scalar_value;
//     }
//     return 0;
// }

// Versao com Intel Intrinsics 
int scalar_matrix_mult(float scalar_value, matrix *m, matrix *r)
{
    if (!m || !r || !m->values || !r->values) return -1;  // Verificação básica

    int total = m->rows * m->cols;
    float *pm = m->values;
    float *pr = r->values;

    int i = 0;

    // Bloco vetorial - processa 8 floats por vez
    __m256 scalar_vec = _mm256_set1_ps(scalar_value);
    for (; i <= total - 8; i += 8) {
        __m256 m_vec = _mm256_loadu_ps(&pm[i]);       // carrega 8 floats da matriz original
        __m256 res_vec = _mm256_mul_ps(m_vec, scalar_vec); // multiplicação vetorial
        _mm256_storeu_ps(&pr[i], res_vec);            // armazena no resultado
    }

    // Resto - processa os elementos restantes (se houver)
    for (; i < total; i++) {
        pr[i] = pm[i] * scalar_value;
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

// int matrix_matrix_mult(matrix *m1, matrix *m2, matrix *r)
// {
//     int rows = m1->rows;
//     int cols = m2->cols;
//     int common = m1->cols;

//     // Inicializa a matriz resultado
//     for (int i = 0; i < rows * cols; i++)
//         r->values[i] = 0;

//     // Multiplicação
//     for (int i = 0; i < rows; i++)
//     {
//         for (int k = 0; k < common; k++)
//         {
//             float m1_val = m1->values[i * common + k];
//             for (int j = 0; j < cols; j++)
//             {
//                 r->values[i * cols + j] += m1_val * m2->values[k * cols + j];
//             }
//         }
//     }

//     return 0;
// }

// Implementacao do matrix_matrix_mult com Intel Intrinsics
int matrix_matrix_mult(matrix *m1, matrix *m2, matrix *r)
{
    if (!m1 || !m2 || !r || !m1->values || !m2->values || !r->values)
        return -1;

    int rows = m1->rows;
    int cols = m2->cols;
    int common = m1->cols;

    // Inicializa matriz resultado
    for (int i = 0; i < rows * cols; i++)
        r->values[i] = 0.0f;

    int j;
    for (int i = 0; i < rows; i++) {
        for (int k = 0; k < common; k++) {
            float a_val = m1->values[i * common + k];
            __m256 a_vec = _mm256_set1_ps(a_val);  // valor de A repetido 8x

            for (j = 0; j <= cols - 8; j += 8) {
                // carrega 8 valores de B[k][j...j+7]
                __m256 b_vec = _mm256_loadu_ps(&m2->values[k * cols + j]);

                // carrega 8 valores atuais de C[i][j...j+7]
                __m256 c_vec = _mm256_loadu_ps(&r->values[i * cols + j]);

                // faz: c_vec += a_vec * b_vec
                c_vec = _mm256_fmadd_ps(a_vec, b_vec, c_vec);  // FMA = multiply-add

                // salva o resultado em C
                _mm256_storeu_ps(&r->values[i * cols + j], c_vec);
            }

            // Resto (caso cols não seja múltiplo de 8)
            for (; j < cols; j++) {
                r->values[i * cols + j] += a_val * m2->values[k * cols + j];
            }
        }
    }

    return 0;
}


// compilacao
// gcc -Wall -o gera-matrix gera_matrix.c
// gcc -Wall -o mlt matrix_lib.c matrix_lib_test.c
//Com instrucao vetorial:
// gcc -Wall -mavx -mfma -o mlt matrix_lib.c matrix_lib_test.c 


// execucao
// ./gera-matrix matrix-test 16000 16000
// ./gera-matrix matrix-test2 16000 16000
// ./gera-matrix matrix-result 16000 16000
// ./gera-matrix matrix-result2 16000 16000
// ./mlt -s 10 -r 16000 -c 16000 -C 16000 -m matrix-test -M matrix-test2 -o matrix-result -O matrix-result2