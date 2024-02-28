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

#define RMSNORM_BLOCK_SIZE 1024
__global__ void thaDNN_s_rmsnorm_kernel(float* o, float* x, float* weight, int size)
{
    int j = threadIdx.x;
    int j_pos = j * 8;
    if (j_pos >= size) return;
    // 768  = 8 * 16 * 6
    // 4096 = 8 * 16 * 32
    // 5120 = 8 * 16 * 40
    __shared__ float local_s[5120];
    float local_x[8];
    float ss = 0;
    for(int i=0 ; i<8 ; ++i)
    {
        local_x[i] = x[j_pos+i];
        ss += local_x[i] * local_x[i];
    }
    local_s[j_pos] = ss;
    __syncthreads();
    
    ss = 0;
    if (j_pos%128==0) // 8 * 16 = 128
    {
        for(int i=j_pos ; i<j_pos+128; i+=8)
            ss += local_s[i];
        local_s[j_pos] = ss;
    }
    __syncthreads();

    ss = 0;
    if (j_pos==0)
    {
        for(int i=0 ; i<size; i+=128)
            ss += local_s[i];
        ss /= size;
        ss += 1e-5f;
        ss = 1.0f / sqrtf(ss);
        local_s[0] = ss;
    }
    __syncthreads();

    ss = local_s[0];
    for(int i=0 ; i<8 ; ++i)
    {
        o[j_pos + i] = weight[j_pos + i] * (ss * local_x[i]);
    }
}

// '_s_' = single persion (float)
// input: o, x, weight allocated on device
// input: size = (768/4096/5120)
thablasStatus_t thaDNN_s_rmsnorm(thablasHandle_t handle, float* o, float* x, float* weight, int size) 
{
    if (size==0 || o == nullptr || x == nullptr || weight == nullptr || handle.current_gpu_id < 0)
    {
        printf("THABLAS RMSNORM ERROR: INVALID ARGUMENT\n"); fflush(stdout);
        return THABLAS_STATUS_ALLOC_FAILED;        
    }

    CHECK_HIP(hipSetDevice(handle.current_gpu_id));
    dim3 blockDim(size / 8);
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
