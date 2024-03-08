#include <hip/hip_runtime.h>
#include <iostream>
/*
* compute the sum of the elements in the input array using shuffle instructions and reduction 
*/

#define WARP_SIZE 64
#define BLOCK_SIZE 128
#define LEN_ARRAY 256 // chúng ta sẽ có 2 block, mỗi block có 128 thread, tổng cộng 256 phần tử

// CHECK_HIP
#define CHECK_HIP(cmd) do {                                 \
  hipError_t error = cmd;                                   \
  if (error != hipSuccess) {                                \
    fprintf(stderr, "error: '%s'(%d) at %s:%d\n",           \
            hipGetErrorString(error), error,                \
            __FILE__, __LINE__);                             \
    exit(EXIT_FAILURE);                                     \
  }                                                         \
} while (0)


__global__ void reduce0(int *g_idata, int *g_odata){
  extern __shared__ int sdata[];

  // each thread loads loads one element from global to shared mem 
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  sdata[tid] = g_idata[i]; // update theo warp size
  __syncthreads();

  // do reduction in shared mem 
  
  for (unsigned int s = 1; s < blockDim.x; s *= 2)
  {
    int index = 2*s*tid;
    if (index < blockDim.x)
    {
      sdata[index] += sdata[index + s];
    }
  }

  // write  result  for this block to global mem 
  if (tid == 0) g_odata[blockIdx.x] = sdata[0];

}

void sum_main(int * h_idata, int * h_odata, int n)
{
  int *g_idata, *g_odata;

  // allocate memory on device 
  CHECK_HIP(hipMalloc((void**)&g_idata, n*sizeof(int)));
  CHECK_HIP(hipMalloc((void**)&g_odata, n*sizeof(int))); 
  CHECK_HIP(hipMemcpy(g_idata, h_idata, n*sizeof(int), hipMemcpyHostToDevice));
  CHECK_HIP(hipMemcpy(g_odata, h_odata, n*sizeof(int), hipMemcpyHostToDevice));

  dim3 blockDim(BLOCK_SIZE);
  dim3 gridDim((n + blockDim.x - 1) / blockDim.x);
  reduce0<<<gridDim, blockDim, n*sizeof(int)>>>(g_idata, g_odata);

  // copy result from device to host
  CHECK_HIP(hipMemcpy(h_odata, g_odata, n*sizeof(int), hipMemcpyDeviceToHost));
  CHECK_HIP(hipDeviceSynchronize());

  // free device memory
  CHECK_HIP(hipFree(g_idata));
  CHECK_HIP(hipFree(g_odata));
}

int main(){
  int *h_idata, *h_odata;
  h_idata = (int *)malloc(LEN_ARRAY * sizeof(int));
  h_odata = (int *)malloc(LEN_ARRAY * sizeof(int));

  for (int i = 0; i < LEN_ARRAY; i++){
    h_idata[i] = 1;
    h_odata[i] = 0;
  }


  sum_main(h_idata, h_odata, LEN_ARRAY);

  for (int i = 0; i < 5; i++){
    std::cout << h_odata[i] << std::endl;
  }

  free(h_idata);
  free(h_odata);
  return 0;
}



