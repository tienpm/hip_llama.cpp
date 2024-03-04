#include "thaDNN.hpp"
#include "thaBLAS.hpp"
#include "hip_helper.hpp"

#include <hip/hip_runtime.h>
#include <omp.h>
#include <hipblas.h>

// size = 1 -> 16384
#define RMS_LOCAL_BANK_SIZE 8
__global__ void thaDNN_s_rmsnorm_kernel(float* o, float* x, float* weight, int size)
{
    int j = threadIdx.x;
    int j_pos = j * 16;
    if (j_pos >= size) return;
    // 768  = 16 * 16 * 3
    // 4096 = 16 * 16 * 16
    // 5120 = 16 * 16 * 20
    // 8192 = 16 * 16 * 32 
    extern __shared__ float local_s[];
    float local_x[16];
    float ss = 0;
    for(int i=0 ; i<16 ; ++i)
    {
        if (j_pos+i < size)
        {
            local_x[i] = x[j_pos+i];
            ss += local_x[i] * local_x[i];
        }
    }
    local_s[j * RMS_LOCAL_BANK_SIZE] = ss;
    __syncthreads();
    
    ss = 0;
    if (j%16==0)
    {
        int i_last = std::min((int)blockDim.x, j+16);
        for(int i=j ; i<i_last; ++i)
            ss += local_s[i * RMS_LOCAL_BANK_SIZE];
        local_s[j * RMS_LOCAL_BANK_SIZE] = ss;
        // printf("j: %d\n", j);
    }
    __syncthreads();

    ss = 0;
    if (j==0)
    {
        int i_last = blockDim.x;
        for(int i=0 ; i<i_last; i+=16)
        {
            ss += local_s[i * RMS_LOCAL_BANK_SIZE];
            // printf("ss i: %f %d\n", ss, i);
        }
        ss /= size;
        ss += 1e-5f;
        ss = 1.0f / sqrtf(ss);
        local_s[0] = ss;
    }
    __syncthreads();

    ss = local_s[0];
    for(int i=0 ; i<16 ; ++i)
        if (j_pos + i < size)
        {
            o[j_pos + i] = weight[j_pos + i] * (ss * local_x[i]);
        }
}

// '_s_' = single persion (float)
// input: o, x, weight allocated on device
// input: size = 1 -> 16384
thablasStatus_t thaDNN_s_rmsnorm(thablasHandle_t handle, float* o, float* x, float* weight, int size) 
{
    if (size==0 || o == nullptr || x == nullptr || weight == nullptr || handle.current_gpu_id < 0)
    {
        printf("THABLAS RMSNORM ERROR: INVALID ARGUMENT\n"); fflush(stdout);
        return THABLAS_STATUS_ALLOC_FAILED;        
    }

    CHECK_HIP(hipSetDevice(handle.current_gpu_id));
    int num_device_threads = (size + 16 - 1) / 16;
    dim3 blockDim(num_device_threads);
    dim3 gridDim(1);
    int local_mem_size = (num_device_threads * RMS_LOCAL_BANK_SIZE) * sizeof(float);
    hipLaunchKernelGGL(thaDNN_s_rmsnorm_kernel, gridDim, blockDim, local_mem_size, 0, o, x, weight, size);
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

/*
*********************************************************************************************************
* softmax
*********************************************************************************************************
*/

// size = 1 -> 32000
__global__ void thaDNN_s_softmax_kernel(float* output, float* x, int size)
{
    /*
    reduction for max value 
    */

    // split array to blocks ,each block is responsible for 32 elements
    int j = threadIdx.x;
    int j_pos = j * 32;
    if (j_pos >= size) return;

    
    // find max value (for numerical stability)
    __shared__ float local_reduction_max[1000]; // 32000/32 = 1000
    float max_val = x[j_pos];
    float local_x[32];

    #pragma unroll
    for (int i = 0; i < 32; i++)
    {
        if (j_pos + i < size)
        {
            local_x[i] = x[j_pos + i];
            if (local_x[i] > max_val) max_val = local_x[i];
        }
    }

    local_reduction_max[j] = max_val;
    __syncthreads();


    // reduction for max value with step 32
    if (j_pos==0)
    {
        max_val = local_reduction_max[0];
        int i_stop = std::min(1000, (int)blockDim.x);
        #pragma unroll
        for (int i = 0; i < i_stop; i++)
        {
            if (local_reduction_max[i] > max_val) max_val = local_reduction_max[i];
        }
        local_reduction_max[0] = max_val;
    }

    __syncthreads();

    /*
    exp and sum
    */

    
    // split array to blocks ,each block is responsible for 32 elements
    __shared__ float local_reduction_sum[1000]; // 32000/32 = 1000
    float sum = 0.0f;
    max_val = local_reduction_max[0];
    #pragma unroll
    for (int i = 0; i < 32; i++)
    {
        if (j_pos + i < size) 
        {
            local_x[i] = expf(local_x[i] - max_val);
            sum += local_x[i];
        }
    }

    //  reduction for sum with step 32
    local_reduction_sum[j] = sum;
    __syncthreads();

    if (j_pos==0)
    {
        sum = 0.0f;
        int i_stop = std::min(1000, (int)blockDim.x);
        #pragma unroll
        for (int i = 0; i < i_stop; i++)
        {
            sum += local_reduction_sum[i];
        }
        local_reduction_sum[0] = sum;
    }

    __syncthreads();

    /*
    normalize
    */
    sum = local_reduction_sum[0];
    #pragma unroll
    for (int i = 0; i < 32; i++)
    {
        if (j_pos + i < size) 
            output[j_pos + i] = local_x[i] / sum;
    }

}

// _s_ = single persion (float)
// input: output, x allocated on device
// input: size = 32000
thablasStatus_t thaDNN_s_softmax(thablasHandle_t handle, float* output, float* x, int size) 
{
    if (size==0 || output == nullptr || x == nullptr || handle.current_gpu_id < 0)
    {
        printf("THABLAS SOFTMAX ERROR: INVALID ARGUMENT\n"); fflush(stdout);
        return THABLAS_STATUS_ALLOC_FAILED;        
    }

    CHECK_HIP(hipSetDevice(handle.current_gpu_id));
    dim3 blockDim((size + 32 - 1) / 32);
    dim3 gridDim(1);
    // dim3 gridSize((size + SOFTMAX_BLOCK_SIZE - 1) / SOFTMAX_BLOCK_SIZE);
    hipLaunchKernelGGL(thaDNN_s_softmax_kernel, gridDim, blockDim, 0, 0, output, x, size);
    CHECK_HIP(hipGetLastError());

    return THABLAS_STATUS_SUCCESS;
}

// _h2d_ = host to device
// [output], [x] are allocated on Host
// only run on 1 devices
// [size] = 1 -> 32000
thablasStatus_t thaDNN_h2d_s_softmax(float* output, float* x, int size) 
{
    if (size==0 || output == nullptr || x == nullptr)
    {
        printf("THABLAS SOFTMAX ERROR: INVALID ARGUMENT\n"); fflush(stdout);
        return THABLAS_STATUS_ALLOC_FAILED;        
    }

    int num_devices;
    CHECK_HIP(hipGetDeviceCount(&num_devices));

    if (!num_devices)
    {
        printf("THABLAS SOFTMAX ERROR: COULD NOT FIND ANY COMPUTE DEVICE\n"); fflush(stdout);
        return THABLAS_STATUS_ALLOC_FAILED;
    }

    float *output_d, *x_d;

    CHECK_HIP(hipSetDevice(0));
    CHECK_HIP(hipMalloc(&output_d, size * sizeof(float)));
    CHECK_HIP(hipMalloc(&x_d, size * sizeof(float)));

    CHECK_HIP(hipMemcpy(x_d, x, size * sizeof(float), hipMemcpyHostToDevice));

    thablasHandle_t handle;
    thablasCreate(&handle);
    thablasStatus_t status = thaDNN_s_softmax(handle, output_d, x_d, size);
    if (status != THABLAS_STATUS_SUCCESS) {
        printf("THABLAS SOFTMAX ERROR: ERROR on Device\n"); fflush(stdout);
    }

    CHECK_HIP(hipMemcpy(output, output_d, size * sizeof(float), hipMemcpyDeviceToHost));

    CHECK_HIP(hipDeviceSynchronize());

    CHECK_HIP(hipFree(output_d));
    CHECK_HIP(hipFree(x_d));

    return THABLAS_STATUS_SUCCESS;
}


