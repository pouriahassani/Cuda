#include <stdio.h>
#include <time.h>
#include <cuda_runtime.h>
const int ARRAY_SIZE = 4;


#define CUDA_CHECK_ERROR(call) \
do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)


// GPU kernel
__global__ void Add(float* a,float* b,float* sum,int ARRAY_SIZE){
    /* to be impelemnted */
}

void add_host(float* a, float*b,float* c){
    for(int i = 0; i < ARRAY_SIZE ; i++){
        c[i] = a[i] + b[i];
    }
}

int main(){
    // Define host arrray pointers and allocate host arrays
    float* a_h = (float *)malloc(ARRAY_SIZE * sizeof(float));
    float* b_h = (float *)malloc(ARRAY_SIZE * sizeof(float)); 
    float* sum_h = (float *)malloc(ARRAY_SIZE * sizeof(float));

    // Define device arrays pointers
    float* a_d;
    float* b_d;
    float* sum_d;
    srand(time(NULL));

    //generate the values for the host input arrays 
    for(int i=0;i<ARRAY_SIZE;i++){
        a_h[i] = rand() / (float)RAND_MAX;;
        b_h[i] = rand() / (float)RAND_MAX;;
    }
        float* c = (float *)malloc(ARRAY_SIZE * sizeof(float));
    add_host(a_h,b_h,c);
    // Allocate memory on device for input and output arrays
    CUDA_CHECK_ERROR(cudaMalloc((void**)&a_d,ARRAY_SIZE * sizeof(float)));
    CUDA_CHECK_ERROR(cudaMalloc((void**)&b_d,ARRAY_SIZE * sizeof(float)));
    CUDA_CHECK_ERROR(cudaMalloc((void**)&sum_d,ARRAY_SIZE * sizeof(float)));

    // copy input arrays from host to device
    CUDA_CHECK_ERROR(cudaMemcpy((void*)a_d,(void* )a_h,ARRAY_SIZE*sizeof(float),cudaMemcpyHostToDevice));
    CUDA_CHECK_ERROR(cudaMemcpy((void*)b_d,(void* )b_h,ARRAY_SIZE*sizeof(float),cudaMemcpyHostToDevice));

    // Create CUDA events
    cudaEvent_t start, stop;
    CUDA_CHECK_ERROR(cudaEventCreate(&start));
    CUDA_CHECK_ERROR(cudaEventCreate(&stop));

    // Record the start event
    CUDA_CHECK_ERROR(cudaEventRecord(start, 0));

    // launch device kernel
    /*To be implemented*/

    // Record the stop event
    CUDA_CHECK_ERROR(cudaEventRecord(stop, 0));

    // Check for kernel launch errors
    CUDA_CHECK_ERROR(cudaGetLastError()); 
    CUDA_CHECK_ERROR(cudaDeviceSynchronize());

    // Calculate elapsed time
    float milliseconds = 0;
    CUDA_CHECK_ERROR(cudaEventElapsedTime(&milliseconds, start, stop));

    // copy the results from device memory to host memory
    CUDA_CHECK_ERROR(cudaMemcpy((void*)sum_h,(void* )sum_d,ARRAY_SIZE*sizeof(float),cudaMemcpyDeviceToHost));

    // Free device memory
    CUDA_CHECK_ERROR(cudaFree(a_d));
    CUDA_CHECK_ERROR(cudaFree(b_d));
    CUDA_CHECK_ERROR(cudaFree(sum_d));


    for(int i=0;i<ARRAY_SIZE;i++){
        if(c[i] != sum_h[i])
            printf("\nwrong answer");
    }
    // Print the time spent by GPU kernel
    printf("Time spent by GPU: %f milliseconds\n", milliseconds);

    // Free host memory
    free(a_h);
    free(b_h);
    free(sum_h);

    return 0;
}