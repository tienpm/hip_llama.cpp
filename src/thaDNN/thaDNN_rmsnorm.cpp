#include "thaDNN/thaDNN_rmsnorm.hpp"

#define WARP_SIZE 64
#define MAX_BLOCK_SIZE 1024

__device__ float warp_reduce_sum(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) 
        val += __shfl_xor(val, offset);
    return val;
}

__device__ float block_reduce_sum(float val) {
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

__global__ void thaDNN_s_rmsnorm_kernel_v2_batch(int n_batches, float* o_batch, float* x_batch, float* weight, int size, int dim) {
    int lx = threadIdx.x;
    int b = blockIdx.x;
    float tmp;
    float ss = 0.0;
    
    float* x = x_batch + b * dim;
    float* o = o_batch + b * dim;
    __shared__ float total_sum;

    for(int i = lx; i < size; i += blockDim.x) {
        tmp = x[i];
        ss += tmp * tmp;
    }

    ss = block_reduce_sum(ss);
    
    if (lx == 0)
    {
        ss /= size;
        ss += 1e-5f;
        ss = 1.0f / sqrtf(ss);
        total_sum = ss;
    }
    __syncthreads();

    ss = total_sum;
    for(int i = lx; i < size; i += blockDim.x) {
        o[i] = weight[i] * (ss * x[i]);
    }
}

thablasStatus_t thaDNN_s_rmsnorm_v2_batch(thablasHandle_t handle, int n_batches, float* o_batch, float* x_batch, float* weight, int size, int dim) {
    // if (size+dim==0 || o_batch == nullptr || x_batch == nullptr || weight == nullptr || handle.current_gpu_id < 0)
    // {
    //     printf("THABLAS RMSNORM V2 BATCH ERROR: INVALID ARGUMENT\n"); fflush(stdout);
    //     return THABLAS_STATUS_ALLOC_FAILED;        
    // }

    // CHECK_HIP(hipSetDevice(handle.current_gpu_id));
    dim3 blockDim(1024);
    dim3 gridDim(n_batches);
    hipLaunchKernelGGL(thaDNN_s_rmsnorm_kernel_v2_batch, 
                       gridDim, blockDim, 0, 0, 
                       n_batches, o_batch, x_batch, weight, size, dim);
    
    // CHECK_HIP(hipGetLastError());
    return THABLAS_STATUS_SUCCESS;
}


// // modify deviceReduceBlockAtomicKernel to caculate sum of squares
// __global__ void thaDNN_s_rmsnorm_kernel_v2(float* o, float* x, float* weight, int size)
// {
//     int lx = threadIdx.x;
//     float tmp;
//     float ss = 0.0;
//     __shared__ float total_sum;
//
//     for(int i = lx; i < size; i += blockDim.x) {
//         tmp = x[i];
//         ss += tmp * tmp;
//     }
//
//     ss = block_reduce_sum(ss);
//     
//     if (lx == 0) {
//         ss /= size;
//         ss += 1e-5f;
//         ss = 1.0f / sqrtf(ss);
//         total_sum = ss;
//     }
//     __syncthreads();
//
//     ss = total_sum;
//     for(int i = lx; i < size; i += blockDim.x) {
//         o[i] = weight[i] * (ss * x[i]);
//     }
// }
//
// // '_s_' = single persion (float)
// // input: o, x, weight allocated on device
// // input: size = 1 -> 16384
// thablasStatus_t thaDNN_s_rmsnorm_v2(thablasHandle_t handle, float* o, float* x, float* weight, int size) 
// {
//     if (size==0 || o == nullptr || x == nullptr || weight == nullptr || handle.current_gpu_id < 0)
//     {
//         printf("THABLAS RMSNORM ERROR: INVALID ARGUMENT\n"); fflush(stdout);
//         return THABLAS_STATUS_ALLOC_FAILED;        
//     }
//
//     CHECK_HIP(hipSetDevice(handle.current_gpu_id));
//     dim3 blockDim(1024);
//     dim3 gridDim(1);
//     hipLaunchKernelGGL(thaDNN_s_rmsnorm_kernel_v2, gridDim, blockDim, 0, 0, o, x, weight, size);
//     
//     CHECK_HIP(hipGetLastError());
//     return THABLAS_STATUS_SUCCESS;
// }
