#include <stdio.h>
#include <cuda_runtime.h>
__global__ void mykernel(){
  int idx = threadIdx.x;
  printf("\n Hello from thread number %d",idx);
}

int main(){

  mykernel<<<1,2>>>();
  cudaDeviceSynchronize();
  return 0;
}