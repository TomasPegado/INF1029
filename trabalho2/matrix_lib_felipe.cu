//Felipe Fortini Franco - 2220501

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "matrix_lib.h"

__global__ void scalar_matrix_mult_kernel(float scalar, float *input, float *output, int total_elements) {
    
    int pos = blockIdx.x * blockDim.x + threadIdx.x;

    if (pos < total_elements) 
    {
        output[pos] = input[pos] * scalar;
    }
}

__global__ void matrix_matrix_mult_kernel(const float *A, const float *B, float *C, int rowsA, int colsA, int colsB) {

    int linha = blockIdx.y * blockDim.y + threadIdx.y;
    int coluna = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (linha < rowsA && coluna < colsB) 
    {
        float sum = 0.0f;
        for (int k = 0; k < colsA; ++k) 
        {
            sum += A[linha * colsA + k] * B[k * colsB + coluna];
        }
        
        C[linha * colsB + coluna] = sum;
    }
}

int scalar_matrix_mult(float scalar_value, matrix *m, matrix *r) {

    if (!m || !r || !m->values || !r->values) 
    {
        return -1;
    }

    unsigned long int tot = m->rows * m->cols;

    if (tot != r->rows * r->cols || m->rows != r->rows || m->cols != r->cols) 
    {
        return -1;
    }

    float *d_in = NULL, *d_out = NULL;
    cudaError_t err;

    if ((err = cudaMalloc((void**)&d_in, tot * sizeof(float))) != cudaSuccess) 
    {
        return -2;
    }
    
    if ((err = cudaMalloc((void**)&d_out, tot * sizeof(float))) != cudaSuccess) 
    { 
        cudaFree(d_in); 
        return -2;
    }

    if ((err = cudaMemcpy(d_in, m->values, tot * sizeof(float), cudaMemcpyHostToDevice)) != cudaSuccess) 
    { 
        cudaFree(d_in); cudaFree(d_out); 
        return -2; 
    }

    int threads = threadsPerBlock > 0 ? threadsPerBlock : 256;
    int blocos = (tot + threads - 1) / threads;

    if (blocos < 1) blocos = 1;

    scalar_matrix_mult_kernel<<<blocos, threads>>>(scalar_value, d_in, d_out, tot);
    cudaDeviceSynchronize();

    if ((err = cudaMemcpy(r->values, d_out, tot * sizeof(float), cudaMemcpyDeviceToHost)) != cudaSuccess) 
    { 
        cudaFree(d_in); 
        cudaFree(d_out); 
        return -2; 
    }
    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}

int matrix_matrix_mult(matrix *m1, matrix *m2, matrix *r) {

    if (!m1 || !m2 || !r || !m1->values || !m2->values || !r->values) 
    {
        return -1;
    }

    if (m1->cols != m2->rows) 
    {
        return -1;
    }

    if (m1->rows != r->rows || m2->cols != r->cols) 
    {
        return -1;
    }

    unsigned long int rowsA = m1->rows, colsA = m1->cols, colsB = m2->cols;
    unsigned long int totA = rowsA * colsA;
    unsigned long int totB = colsA * colsB;
    unsigned long int totC = rowsA * colsB;

    float *d_A = NULL, *d_B = NULL, *d_C = NULL;
    cudaError_t err;

    if ((err = cudaMalloc((void**)&d_A, totA * sizeof(float))) != cudaSuccess) 
    {
        return -2;
    }

    if ((err = cudaMalloc((void**)&d_B, totB * sizeof(float))) != cudaSuccess) { cudaFree(d_A); 
    {
        return -2; }
    }

    if ((err = cudaMalloc((void**)&d_C, totC * sizeof(float))) != cudaSuccess) { cudaFree(d_A); cudaFree(d_B); 
    {
        return -2; }
    }

    if ((err = cudaMemcpy(d_A, m1->values, totA * sizeof(float), cudaMemcpyHostToDevice)) != cudaSuccess) 
    { 
        cudaFree(d_A); 
        cudaFree(d_B); 
        cudaFree(d_C); 
        return -2; 
    }

    if ((err = cudaMemcpy(d_B, m2->values, totB * sizeof(float), cudaMemcpyHostToDevice)) != cudaSuccess) 
    { 
        cudaFree(d_A); 
        cudaFree(d_B); 
        cudaFree(d_C); 
        return -2; 
    }

    int threads = threadsPerBlock > 0 ? threadsPerBlock : 16;
    int blockDimX = (threads > 32) ? 32 : threads; // Limita blockDim.x
    int blockDimY = threads / blockDimX;

    dim3 block(blockDimX, blockDimY);
    dim3 grid((colsB + block.x - 1) / block.x, (rowsA + block.y - 1) / block.y);

    matrix_matrix_mult_kernel<<<grid, block>>>(d_A, d_B, d_C, rowsA, colsA, colsB);
    cudaDeviceSynchronize();

    if ((err = cudaMemcpy(r->values, d_C, totC * sizeof(float), cudaMemcpyDeviceToHost)) != cudaSuccess) 
    { 
        cudaFree(d_A); 
        cudaFree(d_B); 
        cudaFree(d_C); 
        return -2; 
    }
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 0;
}
