#include <stdio.h>
#include <time.h>
#include <cuda_runtime.h>

const int N = 1024;


#define CUDA_CHECK_ERROR(call) \
do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

__global__ void matmul(float* d_A, float* d_B, float* d_C, const int N){
/*To be implemented*/
}


int main(){
    
    // generate the inout array on the host
    float *h_A = (float*)malloc(N * N * sizeof(float));
    float *h_B = (float*)malloc(N * N * sizeof(float));
    srand(time(NULL));
    for(int i=0 ; i< N ; i++){
        for(int j=0 ; j< N ; j++){
            h_A[i * N + j] = rand() / (float)RAND_MAX;
            h_B[i * N + j] = rand() / (float)RAND_MAX;
        }
        
    }
    

    float* h_C = (float*)malloc(N * N * sizeof(float));

    // declare GPU memory pointers
    float* d_A;
    float* d_B;
    float* d_C;
    

    // allocate GPU memory
    cudaMalloc((void**)&d_A , N * N * sizeof(float));
    cudaMalloc((void**)&d_B , N * N * sizeof(float));
    cudaMalloc((void**)&d_C , N * N * sizeof(float));

    // transfer the array to GPU
    cudaMemcpy(d_A, h_A, N * N * sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * N * sizeof(float),cudaMemcpyHostToDevice);

    // Create CUDA events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record the start event
    cudaEventRecord(start, 0);
    // launch kernel
    /*To be implemented*/
    
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
    cudaMemcpy(h_C , d_C , N * N * sizeof(float),cudaMemcpyDeviceToHost);

    // free GPU memory allocation
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Print the time spent by GPU kernel
    printf("Time spent by GPU: %f milliseconds\n", milliseconds);

    
    return 0;
}