#include "hip_helper.hpp"
#include "thaBLAS.hpp"
#include "utils.hpp"

#include <alloca.h>
#include <assert.h>
#include <chrono>

// _s_ = single percision
// _h2d_ = host to device
// all input are allocated on host
thablasStatus_t thaBLAS_h2d_s_matmul(int m, int n, int k, float* A, float* B, float* C, int max_num_gpus = MAX_NUM_SUPPORTED_GPUS)
{
    if (m==0 || n==0 || k==0 || A == nullptr || B == nullptr || C == nullptr || max_num_gpus < 1)
    {
        printf("THABLAS ERROR: INVALID ARGUMENT\n"); fflush(stdout);
        return THABLAS_STATUS_ALLOC_FAILED;        
    }

    int num_gpus;
    CHECK_HIP(hipGetDeviceCount(&num_gpus));

    if (!num_gpus)
    {
        printf("THABLAS ERROR: COULD NOT FIND ANY GPU\n");  fflush(stdout);
        return THABLAS_STATUS_ALLOC_FAILED;
    }

    num_gpus = std::min(num_gpus, max_num_gpus);

    int g_start[num_gpus];
    int g_end[num_gpus];
    int g_m[num_gpus];
    // #pragma omp parallel for num_threads(num_gpus)
    for(int gid = 0 ; gid < num_gpus ; ++gid)
    {
        g_start[gid] = m / num_gpus * (gid+0) + std::min(gid+0, m % num_gpus);
        g_end[gid]   = m / num_gpus * (gid+1) + std::min(gid+1, m % num_gpus);
        g_m[gid]     = g_end[gid] - g_start[gid];
        // printf("g_m[%d]: %d\n", gid, g_m[gid]); fflush(stdout);
    }


    float *A_gpu[num_gpus];
    float *B_gpu[num_gpus];
    float *C_gpu[num_gpus];
    
    #pragma omp parallel for num_threads(num_gpus)
    for(int gid = 0 ; gid < num_gpus ; ++gid)
    {
        CHECK_HIP(hipSetDevice(gid));

        CHECK_HIP(hipMalloc(&A_gpu[gid], g_m[gid] * k * sizeof(float)));
        CHECK_HIP(hipMalloc(&B_gpu[gid],        k * n * sizeof(float)));
        CHECK_HIP(hipMalloc(&C_gpu[gid], g_m[gid] * n * sizeof(float)));

        // hipStream_t upload_stream[num_gpus];
        // CHECK_HIP(hipStreamCreate(&upload_stream[gid]));
    
        int A_offset = g_start[gid] * k;
        // CHECK_HIP(hipMemcpyAsync(A_gpu[gid], A + A_offset, g_m[gid] * k * sizeof(float), hipMemcpyHostToDevice, upload_stream[gid]));
        // CHECK_HIP(hipMemcpyAsync(B_gpu[gid], B, k * n  * sizeof(float), hipMemcpyHostToDevice, upload_stream[gid]));
        CHECK_HIP(hipMemcpy(A_gpu[gid], A + A_offset, g_m[gid] * k * sizeof(float), hipMemcpyHostToDevice));
        CHECK_HIP(hipMemcpy(B_gpu[gid], B, k * n  * sizeof(float), hipMemcpyHostToDevice));

        thablasHandle_t handle;
        thablasCreate(&handle);
        thablasStatus_t status = thaBLAS_s_matmul(handle, g_m[gid], n, k, A_gpu[gid], B_gpu[gid], C_gpu[gid]);
        if (status != THABLAS_STATUS_SUCCESS) {
            printf("THABLAS ERROR: ERROR on Device %d\n", gid); fflush(stdout);
        }
        // dim3 blockDim(GEMM_BLOCK_DIM_X, GEMM_BLOCK_DIM_Y);
        // dim3 gridDim((n + GEMM_BLOCK_DIM_X - 1) / GEMM_BLOCK_DIM_X, (m + GEMM_BLOCK_DIM_Y - 1) / GEMM_BLOCK_DIM_Y);
        // hipLaunchKernelGGL(thablas_Sgemm_kernel, gridDim, blockDim, 0, 0, g_m[gid], n, k, A_gpu[gid], B_gpu[gid], C_gpu[gid]);
        // CHECK_HIP(hipGetLastError());

        int C_offset = g_start[gid] * n;
        CHECK_HIP(hipMemcpy(C + C_offset, C_gpu[gid], g_m[gid] * n * sizeof(float), hipMemcpyDeviceToHost));

        CHECK_HIP(hipDeviceSynchronize());

        CHECK_HIP(hipFree(A_gpu[gid]));
        CHECK_HIP(hipFree(B_gpu[gid]));
        CHECK_HIP(hipFree(C_gpu[gid]));
    }

    return THABLAS_STATUS_SUCCESS;
}

int main() {
  // TODO: Unitetst operator function
  
  return 0;
}
