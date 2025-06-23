// Bruno Wolf - 2212576
// Tomás Lenzi - 2220711


#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h> // AVX intrinsics

#include "matrix_lib.h"

/**
 * @brief Multiplica todos os elementos de uma matriz por um escalar (versão com Intel Intrinsics).
 *
 * Esta função utiliza instruções vetoriais AVX para multiplicar cada elemento da matriz m
 * pelo valor escalar scalar_value e armazena o resultado na matriz r.
 *
 * @param scalar_value Valor escalar para multiplicação.
 * @param m Ponteiro para a matriz de entrada.
 * @param r Ponteiro para a matriz de resultado.
 * @return 0 em caso de sucesso, -1 se ponteiros inválidos.
 */
// int scalar_matrix_mult(float scalar_value, matrix *m, matrix *r)
// {
//     if (!m || !r || !m->values || !r->values)
//         return -1; // Verificação básica

//     int total = m->rows * m->cols;
//     float *pm = m->values;
//     float *pr = r->values;

//     int i = 0;

//     // Bloco vetorial - processa 8 floats por vez
//     __m256 scalar_vec = _mm256_set1_ps(scalar_value);
//     for (; i <= total - 8; i += 8)
//     {
//         __m256 m_vec = _mm256_loadu_ps(&pm[i]);            
//         __m256 res_vec = _mm256_mul_ps(m_vec, scalar_vec); 
//         _mm256_storeu_ps(&pr[i], res_vec);                 
//     }

//     // Resto - processa os elementos restantes (se houver)
//     for (; i < total; i++)
//     {
//         pr[i] = pm[i] * scalar_value;
//     }

//     return 0;
// }

/**
 * @brief Multiplica todos os elementos de uma matriz por um escalar (versão mais rápida).
 *
 * Esta função multiplica cada elemento da matriz m pelo valor escalar scalar_value
 * e armazena o resultado na matriz r. Utiliza acesso sequencial à memória para melhor desempenho.
 *
 * @param scalar_value Valor escalar para multiplicação.
 * @param m Ponteiro para a matriz de entrada.
 * @param r Ponteiro para a matriz de resultado.
 * @return 0 em caso de sucesso.
 */
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

/**
 * @brief Multiplica todos os elementos de uma matriz por um escalar (percorrendo linha por linha).
 *
 * Esta função multiplica cada elemento da matriz m pelo valor escalar scalar_value
 * e armazena o resultado na matriz r. Percorre a matriz linha por linha.
 *
 * @param scalar_value Valor escalar para multiplicação.
 * @param m Ponteiro para a matriz de entrada.
 * @param r Ponteiro para a matriz de resultado.
 * @return 0 em caso de sucesso.
 */
// int scalar_matrix_mult(float scalar_value, matrix *m, matrix *r) {
//     for (int i = 0; i < m->rows; i++) {
//         for (int j = 0; j < m->cols; j++) {
//             r->values[i * r->cols + j] = m->values[i * m->cols + j] * scalar_value;
//         }
//     }
//     return 0;
// }

/**
 * @brief Multiplica todos os elementos de uma matriz por um escalar (percorrendo coluna por coluna).
 *
 * Esta função multiplica cada elemento da matriz m pelo valor escalar scalar_value
 * e armazena o resultado na matriz r. Percorre a matriz coluna por coluna.
 *
 * @param scalar_value Valor escalar para multiplicação.
 * @param m Ponteiro para a matriz de entrada.
 * @param r Ponteiro para a matriz de resultado.
 * @return 0 em caso de sucesso.
 */
// int scalar_matrix_mult(float scalar_value, matrix *m, matrix *r) {
//     for (int j = 0; j < m->cols; j++) {
//         for (int i = 0; i < m->rows; i++) {
//             r->values[i * r->cols + j] = m->values[i * m->cols + j] * scalar_value;
//         }
//     }
//     return 0;
// }

/**
 * @brief Multiplica duas matrizes usando algoritmo otimizado para localidade de cache.
 *
 * Esta função implementa a multiplicação de matrizes com loops reorganizados para
 * melhorar a localidade de cache, percorrendo a matriz m1 linha por linha e a matriz m2
 * de forma a minimizar cache misses.
 *
 * @param m1 Ponteiro para a primeira matriz.
 * @param m2 Ponteiro para a segunda matriz.
 * @param r Ponteiro para a matriz de resultado.
 * @return 0 em caso de sucesso.
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

/**
 * @brief Multiplica duas matrizes usando o algoritmo convencional.
 *
 * Esta função implementa o algoritmo clássico de multiplicação de matrizes,
 * com três loops aninhados na ordem i, j, k. Não faz otimizações de acesso à memória.
 *
 * @param m1 Ponteiro para a primeira matriz.
 * @param m2 Ponteiro para a segunda matriz.
 * @param r Ponteiro para a matriz de resultado.
 * @return 0 em caso de sucesso.
 */
// int matrix_matrix_mult(matrix *m1, matrix *m2, matrix *r)
// {
//     int rows = m1->rows;
//     int cols = m2->cols;
//     int common = m1->cols;

//     for (int i = 0; i < rows; i++) {
//         for (int j = 0; j < cols; j++) {
//             r->values[i * cols + j] = 0.0f; 
//             for (int k = 0; k < common; k++) {
//                 r->values[i * cols + j] += m1->values[i * common + k] * m2->values[k * cols + j];
//             }
//         }
//     }
//     return 0;
// }

/**
 * @brief Multiplica duas matrizes usando Intel Intrinsics (AVX/FMA).
 *
 * Esta função utiliza instruções vetoriais para multiplicar as matrizes m1 e m2,
 * armazenando o resultado em r. É otimizada para processadores com suporte a AVX/FMA.
 *
 * @param m1 Ponteiro para a primeira matriz.
 * @param m2 Ponteiro para a segunda matriz.
 * @param r Ponteiro para a matriz de resultado.
 * @return 0 em caso de sucesso, -1 se ponteiros inválidos.
 */
// int matrix_matrix_mult(matrix *m1, matrix *m2, matrix *r)
// {
//     if (!m1 || !m2 || !r || !m1->values || !m2->values || !r->values)
//         return -1;

//     int rows = m1->rows;
//     int cols = m2->cols;
//     int common = m1->cols;

//     // Inicializa matriz resultado
//     for (int i = 0; i < rows * cols; i++)
//         r->values[i] = 0.0f;

//     int j;
//     for (int i = 0; i < rows; i++)
//     {
//         for (int k = 0; k < common; k++)
//         {
//             float a_val = m1->values[i * common + k];
//             __m256 a_vec = _mm256_set1_ps(a_val); // valor de A repetido 8x

//             for (j = 0; j <= cols - 8; j += 8)
//             {
//                 // carrega 8 valores de B[k][j...j+7]
//                 __m256 b_vec = _mm256_loadu_ps(&m2->values[k * cols + j]);

//                 // carrega 8 valores atuais de C[i][j...j+7]
//                 __m256 c_vec = _mm256_loadu_ps(&r->values[i * cols + j]);

//                 // faz: c_vec += a_vec * b_vec
//                 c_vec = _mm256_fmadd_ps(a_vec, b_vec, c_vec); // FMA = multiply-add

//                 // salva o resultado em C
//                 _mm256_storeu_ps(&r->values[i * cols + j], c_vec);
//             }

//             // Resto (caso cols não seja múltiplo de 8)
//             for (; j < cols; j++)
//             {
//                 r->values[i * cols + j] += a_val * m2->values[k * cols + j];
//             }
//         }
//     }

//     return 0;
// }

/**
 * @brief Instruções para compilação e execução do programa.
 * 
 * Compilação:
 * gcc -Wall -o gera-matrix gera_matrix.c
 * gcc -Wall -o mlt matrix_lib.c matrix_lib_test.c
 * 
 * Com instruções vetoriais:
 * gcc -Wall -mavx -mfma -o mlt matrix_lib.c matrix_lib_test.c
 * 
 * Execução:
 * ./gera-matrix matrix-test 1280 1280
 * ./gera-matrix matrix-test2 1280 1280
 * ./gera-matrix matrix-result 1280 1280
 * ./gera-matrix matrix-result2 1280 1280
 * ./mlt -s 100 -r 1280 -c 1280 -C 1280 -m matrix-test -M matrix-test2 -o matrix-result -O matrix-result2
 */