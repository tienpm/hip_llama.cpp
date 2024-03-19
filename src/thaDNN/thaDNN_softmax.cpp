#include "thaDNN/thaDNN_softmax.hpp"
#include <cstdio>

#define WARP_SIZE 64
#define MAX_BLOCK_SIZE 1024

__device__ float warp_reduce_sum(float val)
{
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) 
        val += __shfl_xor(val, offset);
    return val;
}

__device__ float block_reduce_sum(float val) 
{
    static __shared__ float shared[MAX_BLOCK_SIZE / WARP_SIZE]; 
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;

    val = warp_reduce_sum(val); 

    if (lane == 0)
        shared[wid] = val; 

    __syncthreads(); 

    val = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared[lane] : 0;

    if (wid == 0)
        val = warp_reduce_sum(val); 

    return val;
}

__device__ float warp_reduce_max(float val) {
  for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
    val = std::max(val, __shfl_xor(val, offset));
  return val;
}

__device__ float block_reduce_max(float val) {
  static __shared__ float shared[MAX_BLOCK_SIZE / WARP_SIZE];
  int lane = threadIdx.x % WARP_SIZE;
  int wid = threadIdx.x / WARP_SIZE;

  val = warp_reduce_max(val);

  if (lane == 0) 
    shared[wid] = val;

  __syncthreads();

  val = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared[lane] : -3.402e+38;

  if (wid == 0) val = warp_reduce_max(val);

  return val;
}

__global__ void thaDNN_s_softmax_v2_kernel(float* x, int size) {
    int lx = threadIdx.x;
    int bDim = blockDim.x;
    
    float private_max_val = -3.402e+38;
    __shared__ float max_val;
    for (int i=lx ; i<size ; i+=bDim) {
        private_max_val = std::max(private_max_val, x[i]);
    }

    private_max_val = block_reduce_max(private_max_val);
    if (lx==0) {
        max_val = private_max_val;
    }
    __syncthreads();
    private_max_val = max_val;
    
    float private_sum = 0.0f, tmp;
    __shared__ float sum;
    for (int i =lx; i<size ; i+=bDim) {
        tmp = expf(x[i] - private_max_val);
        x[i] = tmp;
        private_sum += tmp;
    }

    private_sum = block_reduce_sum(private_sum);
    if (lx==0) {
        sum = private_sum;
    }
    __syncthreads();
    private_sum = sum;

    for (int i =lx; i<size ; i+=bDim) {
        x[i] /= private_sum;
    }
}

// _s_ = single persion (float)
// input: output, x allocated on device
// input: size = 32000
thablasStatus_t thaDNN_s_softmax_v2(thablasHandle_t handle, float* x, int size) {
    if (size==0 || x == nullptr || handle.current_gpu_id < 0)
    {
        fprintf(stderr, "THABLAS SOFTMAX ERROR: INVALID ARGUMENT\n");
        return THABLAS_STATUS_ALLOC_FAILED;        
    }

    CHECK_HIP(hipSetDevice(handle.current_gpu_id));
    
    dim3 blockDim(1024);
    dim3 gridDim(1);

    hipLaunchKernelGGL(thaDNN_s_softmax_v2_kernel, gridDim, blockDim, 0, 0, x, size);
    CHECK_HIP(hipGetLastError());
    return THABLAS_STATUS_SUCCESS;
}
