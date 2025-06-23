#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cuda_runtime.h>

#define timedifference_msec(t0,t1) (((t1).tv_sec - (t0).tv_sec) * 1000.0f + ((t1).tv_usec - (t0).tv_usec) / 1000.0f)

#define DATASET_SIZE 1024000

#define THREADS_PER_BLOCK 256
#define MAX_BLOCKS_PER_GRID 4096

typedef float data_t;

__global__ void transmogrifa(int size, data_t *p);

int main(void) {
    data_t *ph;  // ponteiro para o vetor no host
    data_t *pd;  // ponteiro para o vetor no device
    data_t *phFinal;
    struct timeval  start;  // horário inicial
    struct timeval  stop;   // horário final
    cudaError_t cudaError;  // número do erro na GPU

    // alocar vetor no host
    if(!(ph = (data_t *)malloc(DATASET_SIZE * sizeof(data_t)))) {
        fprintf(stderr, "Erro ao tentar alocar memória no host\n");
        exit(1);
    }
    for(int i=0; i<DATASET_SIZE; i++) {
        ph[i] = ((data_t)rand()) / RAND_MAX;
    }

    // aloca vetor no device
    if((cudaError = cudaMalloc(&pd, DATASET_SIZE*sizeof(data_t))) != cudaSuccess) {
        fprintf(stderr, "Erro ao tentar alocar memória no device\n");
        fprintf(stderr, "cudaMalloc returned error %s (code %d)\n", cudaGetErrorString(cudaError), cudaError);
        free(ph);
        exit(2);
    }

    gettimeofday(&start, NULL);
    // transferir dados do host para o device
    if((cudaError = cudaMemcpy(pd, ph, DATASET_SIZE*sizeof(data_t), cudaMemcpyHostToDevice)) != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy (h -> d) returned error %s (code %d), line(%d)\n", 
            cudaGetErrorString(cudaError), cudaError, __LINE__);
        free(ph);
        cudaFree(pd);
		exit(3);
	}

    // realiza a computação na GPU
	int blockSize = THREADS_PER_BLOCK;
	int numBlocks = (DATASET_SIZE + blockSize - 1) / blockSize;
	if (numBlocks > MAX_BLOCKS_PER_GRID) numBlocks = MAX_BLOCKS_PER_GRID;
	transmogrifa<<<numBlocks, blockSize>>>(DATASET_SIZE, pd);
    cudaDeviceSynchronize();    // espera que as threads terminem

    // transferir dados do device para o host
    // alocar vetor no host
    if(!(phFinal = (data_t *)malloc(DATASET_SIZE * sizeof(data_t)))) {
        fprintf(stderr, "Erro ao tentar alocar memória no host\n");
        exit(1);
    }
    if((cudaError = cudaMemcpy(phFinal, pd, DATASET_SIZE*sizeof(data_t), cudaMemcpyDeviceToHost)) != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy (d -> h) returned error %s (code %d), line(%d)\n", 
            cudaGetErrorString(cudaError), cudaError, __LINE__);
        free(ph);
        cudaFree(pd);
		exit(3);
	}

    gettimeofday(&stop, NULL);
    printf("Tempo total na GPU foi de %f ms\n", timedifference_msec(start, stop));

    // realiza a computação na CPU
    data_t *p = ph;
    gettimeofday(&start, NULL);
    for(int i=0; i<DATASET_SIZE; i++) {
        *p = sin(1.0/ *p);
        p++;
    }
    gettimeofday(&stop, NULL);
    printf("Tempo total na CPU foi de %f ms\n", timedifference_msec(start, stop));

    return 0;
}

__global__ void transmogrifa(int size, data_t *p) {
    int nThreads = gridDim.x * blockDim.x;  // número de threads criadas
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;  // número da thread
    int qtd = (size + nThreads - 1) / nThreads;
    int posicao;
    int inicio;

    inicio = qtd * threadId;
    for(int i=0; i<qtd; i++) {
        posicao = inicio + i;
        if(posicao < size) {
            *p = sin(1.0/ *p);
            p++;
        }
    }    
}