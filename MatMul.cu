#include <stdio.h>
#include <time.h>
#include <cuda_runtime.h>

const int ROW_MAX = 1024;
const int COL_MAX = 1024;
#define BLOCK_SIZE 1024

#define CUDA_CHECK_ERROR(call) \
do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

__global__ void square(float* d_A, float* d_B, float* d_C, const int N){
    int C_idx = blockIdx.x * N + threadIdx.x;
    d_C[C_idx] = 0;
    for(int i = 0; i < N ; i++){
        // printf("\nkernel %d %d %d",C_idx,blockIdx.x * N + i,i * N + threadIdx.x);
        d_C[C_idx] += d_A[blockIdx.x * N + i] * d_B[i * N + threadIdx.x];
    }
}

int main(){
    dim3 grid(ROW_MAX,1,1);
    dim3 block(BLOCK_SIZE,1,1);
    
    // generate the inout array on the host
    float *h_A = (float*)malloc(ROW_MAX * COL_MAX * sizeof(float));
    float *h_B = (float*)malloc(ROW_MAX * COL_MAX * sizeof(float));
    srand(time(NULL));
    for(int i=0 ; i< ROW_MAX ; i++){
        for(int j=0 ; j< COL_MAX ; j++){
            h_A[i * COL_MAX + j] = rand() / (float)RAND_MAX;
            h_B[i * COL_MAX + j] = rand() / (float)RAND_MAX;
        }
        
    }
    

    float* h_C = (float*)malloc(ROW_MAX * COL_MAX * sizeof(float));

    // declare GPU memory pointers
    float* d_A;
    float* d_B;
    float* d_C;
    

    // allocate GPU memory
    cudaMalloc((void**)&d_A , ROW_MAX * COL_MAX * sizeof(float));
    cudaMalloc((void**)&d_B , ROW_MAX * COL_MAX * sizeof(float));
    cudaMalloc((void**)&d_C , ROW_MAX * COL_MAX * sizeof(float));

    // transfer the array to GPU
    cudaMemcpy(d_A, h_A, ROW_MAX * COL_MAX * sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, ROW_MAX * COL_MAX * sizeof(float),cudaMemcpyHostToDevice);

    // Create CUDA events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record the start event
    cudaEventRecord(start, 0);

    // launch kernel
    square<<<grid,block>>>(d_A, d_B ,d_C, ROW_MAX);
    CUDA_CHECK_ERROR(cudaGetLastError()); // Check for kernel launch errors
    cudaDeviceSynchronize();
        // Record the stop event
    cudaEventRecord(stop, 0);

    // Synchronize events
    cudaEventSynchronize(stop);

    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // copy back results from GPU memory to CPU  memory
    cudaMemcpy(h_C , d_C , ROW_MAX * COL_MAX * sizeof(float),cudaMemcpyDeviceToHost);

    // free GPU memory allocation
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Print the time spent by GPU kernel
    printf("Time spent by GPU: %f milliseconds\n", milliseconds);
    
    return 0;
}