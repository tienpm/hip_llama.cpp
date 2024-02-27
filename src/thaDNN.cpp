#include "thaDNN.hpp"
#include "thaBLAS.hpp"
#include "hip_helper.hpp"

#include <hip/hip_runtime.h>
#include <omp.h>
#include <hipblas.h>

// void rmsnorm(float* o, float* x, float* weight, int size) {
//   // calculate sum of squares
//   float ss = 0.0f;
//   for (int j = 0; j < size; j++) {
//     ss += x[j] * x[j];
//   }
//   ss /= size;

//   ss += 1e-5f;
//   ss = 1.0f / sqrtf(ss);
//   // normalize and scale
//   for (int j = 0; j < size; j++) {
//     o[j] = weight[j] * (ss * x[j]);
//   }
// }

#define RMSNORM_BLOCK_SIZE 768
__global__ void thaDNN_s_rmsnorm_kernel(float* o, float* x, float* weight, int size)
{
    int j = threadIdx.x;
    float xj = x[j];
    float ss = xj;
    ss = ss * ss;

    __shared__ float local_s[RMSNORM_BLOCK_SIZE];
    local_s[j] = ss;
    __syncthreads();

    // 768 = 8 * 8 * 12
    ss = 0;
    if (j%8==0)
    {
        for(int i=j ; i<j+8; ++i)
            ss += local_s[i];
        local_s[j] = ss;
    }
    __syncthreads();

    ss = 0;
    if (j%64==0) // 8 * 8
    {
        for(int i=j ; i<j+64; i+=8)
            ss += local_s[i];
        local_s[j] = ss;
    }
    __syncthreads();

    ss = 0;
    if (j==0)
    {
        for(int i=0 ; i<size; i+=64)
            ss += local_s[i];
        ss /= size;
        ss += 1e-5f;
        ss = 1.0f / sqrtf(ss);
        local_s[0] = ss;
    }
    __syncthreads();

    ss = local_s[0];
    o[j] = weight[j] * (ss * xj);
}

// '_s_' = single persion (float)
// input: o, x, weight allocated on device
// input: size = 768 = 256 * 3
thablasStatus_t thaDNN_s_rmsnorm(thablasHandle_t handle, float* o, float* x, float* weight, int size) 
{
    if (size==0 || o == nullptr || x == nullptr || weight == nullptr || handle.current_gpu_id < 0)
    {
        printf("THABLAS RMSNORM ERROR: INVALID ARGUMENT\n"); fflush(stdout);
        return THABLAS_STATUS_ALLOC_FAILED;        
    }

    CHECK_HIP(hipSetDevice(handle.current_gpu_id));
    dim3 blockDim(RMSNORM_BLOCK_SIZE);
    dim3 gridDim(1);
    // dim3 gridSize((size + RMSNORM_BLOCK_SIZE - 1) / RMSNORM_BLOCK_SIZE);
    hipLaunchKernelGGL(thaDNN_s_rmsnorm_kernel, gridDim, blockDim, 0, 0, o, x, weight, size);
    CHECK_HIP(hipGetLastError());

    return THABLAS_STATUS_SUCCESS;
}

// _h2d_ = host to device
// o, x, weight allocated on Host
// only run on 1 devices
thablasStatus_t thaDNN_h2d_s_rmsnorm(float* o, float* x, float* weight, int size) 
{
    if (size==0 || o == nullptr || x == nullptr || weight == nullptr)
    {
        printf("THABLAS RMSNORM ERROR: INVALID ARGUMENT\n"); fflush(stdout);
        return THABLAS_STATUS_ALLOC_FAILED;        
    }

    int num_devices;
    CHECK_HIP(hipGetDeviceCount(&num_devices));

    if (!num_devices)
    {
        printf("THABLAS RMSNORM ERROR: COULD NOT FIND ANY COMPUTE DEVICE\n"); fflush(stdout);
        return THABLAS_STATUS_ALLOC_FAILED;
    }

    float *o_d, *x_d, *weight_d;

    CHECK_HIP(hipSetDevice(0));
    CHECK_HIP(hipMalloc(&o_d, size * sizeof(float)));
    CHECK_HIP(hipMalloc(&x_d, size * sizeof(float)));
    CHECK_HIP(hipMalloc(&weight_d, size * sizeof(float)));

    CHECK_HIP(hipMemcpy(x_d, x, size * sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(weight_d, weight, size * sizeof(float), hipMemcpyHostToDevice));

    thablasHandle_t handle;
    thablasCreate(&handle);
    thablasStatus_t status = thaDNN_s_rmsnorm(handle, o_d, x_d, weight_d, size);
    if (status != THABLAS_STATUS_SUCCESS) {
        printf("THABLAS RMSNORM ERROR: ERROR on Device\n"); fflush(stdout);
    }

    CHECK_HIP(hipMemcpy(o, o_d, size * sizeof(float), hipMemcpyDeviceToHost));

    CHECK_HIP(hipDeviceSynchronize());

    CHECK_HIP(hipFree(o_d));
    CHECK_HIP(hipFree(x_d));
    CHECK_HIP(hipFree(weight_d));

    return THABLAS_STATUS_SUCCESS;
}
