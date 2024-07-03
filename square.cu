#include <stdio.h>
#include <time.h>
#define ARRAY_SIZE 1024
#include <cuda_runtime.h>
__global__ void square(float* d_out, float* d_in){
    int idx = threadIdx.x;
    d_out[idx] = d_in[idx]*d_in[idx];
}

int main(){
    
    // generate the inout array on the host
    float h_in[ARRAY_SIZE];
    srand(time(NULL));
    for(int i=0 ; i< ARRAY_SIZE ; i++){
        h_in[i] = float(i);
    }

    float h_out[ARRAY_SIZE];

    // declare GPU memory pointers
    float* d_in;
    float* d_out;

    // allocate GPU memory
    cudaMalloc((void**)&d_in,ARRAY_SIZE * sizeof(float));
    cudaMalloc((void**)&d_out,ARRAY_SIZE * sizeof(float));

    // transfer the array to GPU
    cudaMemcpy(d_in, h_in, ARRAY_SIZE*sizeof(float),cudaMemcpyHostToDevice);

    // Create CUDA events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record the start event
    cudaEventRecord(start, 0);

    // launch kernel
    square<<<1,ARRAY_SIZE>>>(d_in,d_out);

        // Record the stop event
    cudaEventRecord(stop, 0);

    // Synchronize events
    cudaEventSynchronize(stop);

    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // copy back results from GPU memory to CPU  memory
    cudaMemcpy(d_out, h_out, ARRAY_SIZE*sizeof(float),cudaMemcpyDeviceToHost);

    // free GPU memory allocation
    cudaFree(d_in);
    cudaFree(d_out);

    // Print the time spent by GPU kernel
    printf("Time spent by GPU: %f milliseconds\n", milliseconds);
    return 0;
}