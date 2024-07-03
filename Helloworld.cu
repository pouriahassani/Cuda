#include <stdio.h>
#include <cuda_runtime.h>
__global__ void mykernel(){
}

int main(){

  mykernel<<<1,1>>>();
  printf("\nCalled GPU kernel!");
  return 0;
}