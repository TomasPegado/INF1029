#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "matrix_lib.h"

// =============================
// KERNELS CUDA (placeholders)
// =============================

// Kernel para multiplicação escalar
__global__ void scalar_matrix_mult_kernel(float scalar, float *input, float *output, int total_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        output[idx] = input[idx] * scalar;
    }
}

// Kernel para multiplicação de matrizes
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

// Outras funções do header podem ser implementadas aqui, se necessário. 