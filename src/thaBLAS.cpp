#include "thaBLAS.hpp"
#include "hip_helper.hpp"

#include <hip/hip_runtime.h>
#include <omp.h>
#include <hipblas.h>

#define GEMM_BLOCK_DIM_X 32
#define GEMM_BLOCK_DIM_Y 4

#define VDS_BLOCK_DIM 32

thablasStatus_t thablasCreate(thablasHandle_t* handle)
{
    int current_gpu_id;
    CHECK_HIP(hipGetDevice(&current_gpu_id));
    handle->current_gpu_id = current_gpu_id;

    return THABLAS_STATUS_SUCCESS;
}

thablasStatus_t thablasDestroy(thablasHandle_t handle)
{
    // 
    return THABLAS_STATUS_SUCCESS;
}



/*
 * ===========================================================================
 *    level 1 BLAS
 * ===========================================================================
 */

__global__ void thablas_Svds_kernel(int n, float *A, float *B, float val)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i>=n) return;

    B[i] = A[i] / val;
}

thablasStatus_t thablas_Svds(thablasHandle_t handle, int n, float* A, float* B, float val)
{
    if (n == 0 || A == nullptr || B == nullptr || val == 0 || handle.current_gpu_id < 0)
    {
        printf("THABLAS ERROR: INVALID ARGUMENT\n");
        return THABLAS_STATUS_ALLOC_FAILED;   
    }

    dim3 blockDim(VDS_BLOCK_DIM);
    dim3 gridDim((n + VDS_BLOCK_DIM - 1) / VDS_BLOCK_DIM);
    // printf("n=%d\n", n); fflush(stdout);
    hipLaunchKernelGGL(thablas_Svds_kernel, gridDim, blockDim, 0, 0, n, A, B, val);
    CHECK_HIP(hipGetLastError());

    return THABLAS_STATUS_SUCCESS;
}

thablasStatus_t thablas_c2d_Svds(int n, float* A, float* B, float val, int max_num_gpus = MAX_NUM_SUPPORTED_GPUS)
{
    if (n == 0 || A == nullptr || B == nullptr || val == 0 || max_num_gpus < 1)
    {
        printf("THABLAS ERROR: INVALID ARGUMENT\n"); fflush(stdout);
        return THABLAS_STATUS_ALLOC_FAILED;   
    }

    int num_gpus;
    CHECK_HIP(hipGetDeviceCount(&num_gpus));

    if (!num_gpus)
    {
        printf("THABLAS ERROR: COULD NOT FIND ANY GPU\n"); fflush(stdout);
        return THABLAS_STATUS_ALLOC_FAILED;
    }
    num_gpus = std::min(num_gpus, max_num_gpus);

    int g_start[num_gpus];
    int g_end[num_gpus];
    int g_n[num_gpus];

    // #pragma omp parallel for num_threads(num_gpus)
    for(int gid = 0 ; gid < num_gpus ; ++gid)
    {
        g_start[gid] = n / num_gpus * (gid+0) + std::min(gid+0, n % num_gpus);
        g_end[gid]   = n / num_gpus * (gid+1) + std::min(gid+1, n % num_gpus);
        g_n[gid]     = g_end[gid] - g_start[gid];
    }

    float *A_gpu[num_gpus];
    float *B_gpu[num_gpus];
    #pragma omp parallel for num_threads(num_gpus)
    for(int gid = 0 ; gid < num_gpus ; ++gid)
    {
        CHECK_HIP(hipSetDevice(gid));
        CHECK_HIP(hipMalloc(&A_gpu[gid], g_n[gid] * sizeof(float)));
        CHECK_HIP(hipMalloc(&B_gpu[gid], g_n[gid] * sizeof(float)));
    }

    #pragma omp parallel for num_threads(num_gpus)
    for(int gid = 0 ; gid < num_gpus ; ++gid)
    {
        CHECK_HIP(hipSetDevice(gid));

        int offset = g_start[gid];
        CHECK_HIP(hipMemcpy(A_gpu[gid], A + offset, g_n[gid] * sizeof(float), hipMemcpyHostToDevice));

        thablasHandle_t handle;
        thablasCreate(&handle);
        thablasStatus_t status = thablas_Svds(handle, g_n[gid], A_gpu[gid], B_gpu[gid], val);
        if (status != THABLAS_STATUS_SUCCESS) {
            printf("THABLAS ERROR: ERROR on Device %d\n", gid); fflush(stdout);
        }

        CHECK_HIP(hipMemcpy(B + offset, B_gpu[gid], g_n[gid] * sizeof(float), hipMemcpyDeviceToHost));

        CHECK_HIP(hipDeviceSynchronize());
    }

    return THABLAS_STATUS_SUCCESS;
}

/*
 * ===========================================================================
 *    level 2 BLAS
 * ===========================================================================
 */

/*
 * ===========================================================================
 *    level 3 BLAS
 * ===========================================================================
 */

__global__ void thablas_Sgemm_kernel(int M, int N,int K, float *A, float *B, float *C)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (j >= N || i >= M) return;

    // printf("%d %d %0.2f\n", i, j, A[i*K + j]);

    float sum = 0;
    for(int ik = 0 ; ik < K ; ++ik)
    {
        // if (i==0 && j==0)
        // {
        //     printf("%0.2f * %0.2f\n", A[i*K + ik], B[ik*N + j]);
        // }
        sum += A[i*K + ik] * B[ik*N + j];
    }
    C[i*N + j] = sum;
}

thablasStatus_t thablas_Sgemm(thablasHandle_t handle, int m, int n, int k, float* A, float* B, float* C)
{
    if (m==0 || n==0 || k==0 || A == nullptr || B == nullptr || C == nullptr || handle.current_gpu_id < 0)
    {
        printf("THABLAS ERROR: INVALID ARGUMENT\n"); fflush(stdout);
        return THABLAS_STATUS_ALLOC_FAILED;        
    }

    CHECK_HIP(hipSetDevice(handle.current_gpu_id));
    dim3 blockDim(GEMM_BLOCK_DIM_X, GEMM_BLOCK_DIM_Y);
    dim3 gridDim((n + GEMM_BLOCK_DIM_X - 1) / GEMM_BLOCK_DIM_X, (m + GEMM_BLOCK_DIM_Y - 1) / GEMM_BLOCK_DIM_Y);

    hipLaunchKernelGGL(thablas_Sgemm_kernel, gridDim, blockDim, 0, 0, m, n, k, A, B, C);
    CHECK_HIP(hipGetLastError());

    return THABLAS_STATUS_SUCCESS;
}

thablasStatus_t thablas_c2d_Sgemm(int m, int n, int k, float* A, float* B, float* C, int max_num_gpus = MAX_NUM_SUPPORTED_GPUS)
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
        thablasStatus_t status = thablas_Sgemm(handle, g_m[gid], n, k, A_gpu[gid], B_gpu[gid], C_gpu[gid]);
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
    }

    return THABLAS_STATUS_SUCCESS;
}


/*

    * ===========================================================================
    *    exponential function in softmax layer
    * ===========================================================================
    
*/


__global__ void thablas_Sexp_kernel(float *vector_in, float *vector_out, int vector_size)
{
    
    /*
    exponential of 1 dimention vector elements
        - input: (N, C)
        - output: (N, C)
    */

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i>=vector_size) return;

    vector_out[i] = exp(vector_in[i]);
}


thablasStatus_t thablas_Sexp(thablasHandle_t handle, float* vector_in, float* vector_out, int vector_size)
{
    if (vector_size == 0 || vector_in == nullptr || vector_out == nullptr || handle.current_gpu_id < 0)
    {
        printf("THABLAS ERROR: INVALID ARGUMENT\n");
        return THABLAS_STATUS_ALLOC_FAILED;   
    }

    dim3 blockDim(VDS_BLOCK_DIM);
    dim3 gridDim((vector_size + VDS_BLOCK_DIM - 1) / VDS_BLOCK_DIM);
    hipLaunchKernelGGL(thablas_Sexp_kernel, gridDim, blockDim, 0, 0, vector_in, vector_out, vector_size);
    CHECK_HIP(hipGetLastError());

    return THABLAS_STATUS_SUCCESS;
}

thablasStatus_t thablas_c2d_Sexp(float* vector_in, float* vector_out, int vector_size, int max_num_gpus = MAX_NUM_SUPPORTED_GPUS)
{
    if (vector_size == 0 || vector_in == nullptr || vector_out == nullptr || max_num_gpus < 1)
    {
        printf("THABLAS ERROR: INVALID ARGUMENT\n"); fflush(stdout);
        return THABLAS_STATUS_ALLOC_FAILED;   
    }

    int num_gpus;
    CHECK_HIP(hipGetDeviceCount(&num_gpus));

    if (!num_gpus)
    {
        printf("THABLAS ERROR: COULD NOT FIND ANY GPU\n"); fflush(stdout);
        return THABLAS_STATUS_ALLOC_FAILED;
    }
    num_gpus = std::min(num_gpus, max_num_gpus);

    int g_start[num_gpus];
    int g_end[num_gpus];
    int g_n[num_gpus];

    // #pragma omp parallel for num_threads(num_gpus)
    for(int gid = 0 ; gid < num_gpus ; ++gid)
    {
        g_start[gid] = vector_size / num_gpus * (gid+0) + std::min(gid+0, vector_size % num_gpus);
        g_end[gid]   = vector_size / num_gpus * (gid+1) + std::min(gid+1, vector_size % num_gpus);
        g_n[gid]     = g_end[gid] - g_start[gid];
    }

    float *vector_in_gpu[num_gpus];
    float *vector_out_gpu[num_gpus];
    #pragma omp parallel for num_threads(num_gpus)
    for(int gid = 0 ; gid < num_gpus ; ++gid)
    {
        CHECK_HIP(hipSetDevice(gid));
        CHECK_HIP(hipMalloc(&vector_in_gpu[gid], g_n[gid] * sizeof(float)));
        CHECK_HIP(hipMalloc(&vector_out_gpu[gid], g_n[gid] * sizeof(float)));
    }

    #pragma omp parallel for num_threads(num_gpus)
    for(int gid = 0 ; gid < num_gpus ; ++gid)
    {
        CHECK_HIP(hipSetDevice(gid));

        int offset = g_start[gid];
        CHECK_HIP(hipMemcpy(vector_in_gpu[gid], vector_in + offset  , g_n[gid] * sizeof(float), hipMemcpyHostToDevice));

        thablasHandle_t handle;
        thablasCreate(&handle);

        thablasStatus_t status = thablas_Sexp(handle, vector_in_gpu[gid], vector_out_gpu[gid], g_n[gid]);
        if (status != THABLAS_STATUS_SUCCESS) {
            printf("THABLAS ERROR: ERROR on Device %d\n", gid); fflush(stdout);
        }

        CHECK_HIP(hipMemcpy(vector_out + offset, vector_out_gpu[gid], g_n[gid] * sizeof(float), hipMemcpyDeviceToHost));

        CHECK_HIP(hipDeviceSynchronize());

    }

    return THABLAS_STATUS_SUCCESS;
}

