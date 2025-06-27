/**
 * @file matrix_lib.cu
 * @brief Implementação de operações matriciais utilizando CUDA.
 *
 * Este arquivo contém kernels CUDA e funções host para multiplicação escalar e multiplicação de matrizes,
 * utilizando aceleração por GPU. As funções são utilizadas para operações de álgebra linear em matrizes
 * representadas pela struct 'matrix'.
 *
 * @author Bruno Wolf
 * @author Tomás Lenzi
 * @date 2024
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "matrix_lib.h"

// =============================
// KERNELS CUDA (placeholders)
// =============================

/**
 * @brief Kernel CUDA para multiplicação escalar de matriz.
 *
 * Multiplica cada elemento do vetor de entrada por um escalar e armazena o resultado no vetor de saída.
 *
 * @param scalar Escalar a ser multiplicado.
 * @param input Vetor de entrada (matriz linearizada).
 * @param output Vetor de saída (matriz linearizada).
 * @param total_elements Número total de elementos na matriz.
 */
__global__ void scalar_matrix_mult_kernel(float scalar, float *input, float *output, int total_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        output[idx] = input[idx] * scalar;
    }
}

/**
 * @brief Kernel CUDA para multiplicação de matrizes.
 *
 * Calcula o produto de duas matrizes A e B e armazena o resultado em C.
 *
 * @param A Matriz A (entrada), de dimensão rowsA x colsA.
 * @param B Matriz B (entrada), de dimensão colsA x colsB.
 * @param C Matriz resultado (saída), de dimensão rowsA x colsB.
 * @param rowsA Número de linhas da matriz A.
 * @param colsA Número de colunas da matriz A (e linhas da matriz B).
 * @param colsB Número de colunas da matriz B.
 */
__global__ void matrix_matrix_mult_kernel(const float *A, const float *B, float *C, int rowsA, int colsA, int colsB) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rowsA && col < colsB) {
        float sum = 0.0f;
        for (int k = 0; k < colsA; ++k) {
            sum += A[row * colsA + k] * B[k * colsB + col];
        }
        C[row * colsB + col] = sum;
    }
}

// =============================
// FUNÇÕES HOST (implementação das funções do header)
// =============================

/**
 * @brief Multiplica uma matriz por um escalar utilizando CUDA.
 *
 * @param scalar_value Valor escalar para multiplicação.
 * @param m Ponteiro para a matriz de entrada.
 * @param r Ponteiro para a matriz de resultado (deve ter as mesmas dimensões de m).
 * @return 0 em caso de sucesso, -1 para erro de parâmetros, -2 para erro de alocação/cópia CUDA.
 */
int scalar_matrix_mult(float scalar_value, matrix *m, matrix *r) {
    if (!m || !r || !m->values || !r->values) return -1;
    unsigned long int total = m->rows * m->cols;
    if (total != r->rows * r->cols || m->rows != r->rows || m->cols != r->cols) return -1;

    float *d_in = NULL, *d_out = NULL;
    cudaError_t err;

    // Aloca memória no device
    if ((err = cudaMalloc((void**)&d_in, total * sizeof(float))) != cudaSuccess) return -2;
    if ((err = cudaMalloc((void**)&d_out, total * sizeof(float))) != cudaSuccess) { cudaFree(d_in); return -2; }

    // Copia dados do host para o device
    if ((err = cudaMemcpy(d_in, m->values, total * sizeof(float), cudaMemcpyHostToDevice)) != cudaSuccess) { cudaFree(d_in); cudaFree(d_out); return -2; }

    // Configura grid/block
    int threads = threadsPerBlock > 0 ? threadsPerBlock : 256;
    int blocks = (total + threads - 1) / threads;
    if (blocks < 1) blocks = 1;

    // Chama kernel
    scalar_matrix_mult_kernel<<<blocks, threads>>>(scalar_value, d_in, d_out, total);
    cudaDeviceSynchronize();

    // Copia resultado do device para o host
    if ((err = cudaMemcpy(r->values, d_out, total * sizeof(float), cudaMemcpyDeviceToHost)) != cudaSuccess) { cudaFree(d_in); cudaFree(d_out); return -2; }

    // Libera memória do device
    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}

/**
 * @brief Multiplica duas matrizes utilizando CUDA.
 *
 * @param m1 Ponteiro para a matriz A (entrada).
 * @param m2 Ponteiro para a matriz B (entrada).
 * @param r Ponteiro para a matriz resultado (deve ter dimensões compatíveis).
 * @return 0 em caso de sucesso, -1 para erro de parâmetros, -2 para erro de alocação/cópia CUDA.
 */
int matrix_matrix_mult(matrix *m1, matrix *m2, matrix *r) {
    if (!m1 || !m2 || !r || !m1->values || !m2->values || !r->values) return -1;
    if (m1->cols != m2->rows) return -1;
    if (m1->rows != r->rows || m2->cols != r->cols) return -1;
    unsigned long int rowsA = m1->rows, colsA = m1->cols, colsB = m2->cols;
    unsigned long int totalA = rowsA * colsA;
    unsigned long int totalB = colsA * colsB;
    unsigned long int totalC = rowsA * colsB;

    float *d_A = NULL, *d_B = NULL, *d_C = NULL;
    cudaError_t err;

    // Aloca memória no device
    if ((err = cudaMalloc((void**)&d_A, totalA * sizeof(float))) != cudaSuccess) return -2;
    if ((err = cudaMalloc((void**)&d_B, totalB * sizeof(float))) != cudaSuccess) { cudaFree(d_A); return -2; }
    if ((err = cudaMalloc((void**)&d_C, totalC * sizeof(float))) != cudaSuccess) { cudaFree(d_A); cudaFree(d_B); return -2; }

    // Copia dados do host para o device
    if ((err = cudaMemcpy(d_A, m1->values, totalA * sizeof(float), cudaMemcpyHostToDevice)) != cudaSuccess) { cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); return -2; }
    if ((err = cudaMemcpy(d_B, m2->values, totalB * sizeof(float), cudaMemcpyHostToDevice)) != cudaSuccess) { cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); return -2; }

    // Configura grid/block (2D)
    int threads = threadsPerBlock > 0 ? threadsPerBlock : 16;
    int blockDimX = (threads > 32) ? 32 : threads; // Limita blockDim.x
    int blockDimY = threads / blockDimX;
    dim3 block(blockDimX, blockDimY);
    dim3 grid((colsB + block.x - 1) / block.x, (rowsA + block.y - 1) / block.y);

    // Chama kernel
    matrix_matrix_mult_kernel<<<grid, block>>>(d_A, d_B, d_C, rowsA, colsA, colsB);
    cudaDeviceSynchronize();

    // Copia resultado do device para o host
    if ((err = cudaMemcpy(r->values, d_C, totalC * sizeof(float), cudaMemcpyDeviceToHost)) != cudaSuccess) { cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); return -2; }

    // Libera memória do device
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 0;
}

//  Compilação CUDA:
// gcc -Wall -o gera-matrix gera_matrix.c
// nvcc -o mlt_gpu matrix_lib.cu matrix_lib_test.cu

// Execução:
// ./gera-matrix floats1.dat 5600 6400
// ./gera-matrix floats2.dat 6400 7200
// ./gera-matrix result1.dat 5600 6400
// ./gera-matrix result2.dat 6400 7200

// Execução CUDA:  
//  ./mlt_gpu -s 10000.0 -r 2400 -c 3200 -C 4000 -m floats1.dat -M floats2.dat -o result1.dat -O result2.dat -t 256 -g 4096