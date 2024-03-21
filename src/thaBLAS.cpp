#include "thaBLAS.hpp"
#include "hip_helper.hpp"

#include <hip/hip_runtime.h>
#include <omp.h>
#include <hipblas.h>

#define WARP_SIZE 64
#define MAX_BLOCK_SIZE 1024
#define GEMM_BLOCK_DIM_X 32
#define GEMM_BLOCK_DIM_Y 4
#define VDS_BLOCK_DIM 32

// #define MATRIX_CORE 1

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

/*
 * ===========================================================================
 *    Init
 * ===========================================================================
 */

thablasStatus_t thablasCreate(thablasHandle_t* handle)
{
    int current_gpu_id;
    CHECK_HIP(hipGetDevice(&current_gpu_id));
    handle->current_gpu_id = current_gpu_id;

    CHECK_HIP(hipStreamCreate(&handle->calc_stream));
    CHECK_HIP(hipStreamCreate(&handle->copy_stream));

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


/*
 * ===========================================================================
 *    level 2 BLAS
 * ===========================================================================
 */

__global__ void thaBLAS_s_vecaddvec_kernel(float *a, float *b, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i<size) 
        a[i] += b[i];
}

thablasStatus_t thaBLAS_s_vecaddvec(thablasHandle_t* handle, float *a, float *b, int size)
{
    // if (a == nullptr || b == nullptr || size == 0)
    // {
    //     printf("THABLAS VEC ADD VEC ERROR: INVALID ARGUMENT\n"); fflush(stdout);
    //     return THABLAS_STATUS_ALLOC_FAILED;        
    // }

    // CHECK_HIP(hipSetDevice(handle.current_gpu_id));
    dim3 blockDim(64);
    dim3 gridDim((size + 64 - 1) / 64);
    hipLaunchKernelGGL(thaBLAS_s_vecaddvec_kernel, gridDim, blockDim, 0, handle->calc_stream, a, b, size);
    // CHECK_HIP(hipGetLastError());

    return THABLAS_STATUS_SUCCESS;
}

/*
 * ===========================================================================
 *    level 3 BLAS
 * ===========================================================================
 */

__global__ void thaBLAS_s_matmul_kernel(int M, int N,int K, float *A, float *B, float *C)
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

// _s_ = single percision
// all input are allocated on device
thablasStatus_t thaBLAS_s_matmul(thablasHandle_t handle, int m, int n, int k, float* A, float* B, float* C)
{
    if (m==0 || n==0 || k==0 || A == nullptr || B == nullptr || C == nullptr || handle.current_gpu_id < 0)
    {
        printf("THABLAS ERROR: INVALID ARGUMENT\n"); fflush(stdout);
        return THABLAS_STATUS_ALLOC_FAILED;        
    }

    CHECK_HIP(hipSetDevice(handle.current_gpu_id));
    dim3 blockDim(GEMM_BLOCK_DIM_X, GEMM_BLOCK_DIM_Y);
    dim3 gridDim((n + GEMM_BLOCK_DIM_X - 1) / GEMM_BLOCK_DIM_X, (m + GEMM_BLOCK_DIM_Y - 1) / GEMM_BLOCK_DIM_Y);

    hipLaunchKernelGGL(thaBLAS_s_matmul_kernel, gridDim, blockDim, 0, 0, m, n, k, A, B, C);
    CHECK_HIP(hipGetLastError());

    return THABLAS_STATUS_SUCCESS;
}

__global__ void thaBLAS_s_matmul_batch_kernel(float *C_batch, float *B_batch, float *A, int K, int M, int Coff, int has_pos, int pos[], int C_batch_size, int B_batch_size) {
    int gx = blockIdx.x;
    int b = blockIdx.y;
    int lx = threadIdx.x;
    float sum = 0.0f;

    float *C = C_batch + Coff + has_pos * pos[b] + b * C_batch_size;
    float *B = B_batch + b * B_batch_size;
    for (int k=lx ; k<K ; k+=blockDim.x)
    {
        sum += A[gx*K + k] * B[k];
    }
    sum = block_reduce_sum(sum);
    if (lx == 0)
    {
        C[gx] = sum;
    }
}

// A[M,K] x B[K,1] = C[M, 1]
thablasStatus_t thaBLAS_s_matmul_batch(thablasHandle_t* handle, int n_batches, float *C_batch, float *B_batch, float *A, int K, int M, int Coff, int has_pos, int pos_d[], int C_batch_size, int B_batch_size) {
    // if (K + M + n_batches==0 || A == nullptr || B_batch == nullptr || C_batch == nullptr || handle.current_gpu_id < 0)
    // {
    //     printf("THABLAS MAT MUL VEC BATCH ERROR: INVALID ARGUMENT\n"); fflush(stdout);
    //     return THABLAS_STATUS_ALLOC_FAILED;        
    // }

    // CHECK_HIP(hipSetDevice(handle.current_gpu_id));
    dim3 blockDim(MAX_BLOCK_SIZE);
    dim3 gridDim(M, n_batches);

    hipLaunchKernelGGL(thaBLAS_s_matmul_batch_kernel, 
                       gridDim, blockDim, 0, 0, 
                       C_batch, B_batch, A, K, M, Coff, has_pos, pos_d, C_batch_size, B_batch_size);
    // CHECK_HIP(hipGetLastError());

    return THABLAS_STATUS_SUCCESS;
}

/*
 * ===========================================================================
 *    level 2 BLAS - Mat multiply vec
 * ===========================================================================
 */

// thablasStatus_t thaBLAS_h2d_s_matmulvec(float *C, float *B, float *A, int K, int M)
// {
//     return thaBLAS_h2d_s_matmul(M, 1, K, A, B, C);
// }

thablasStatus_t thaBLAS_s_matmulvec(thablasHandle_t handle, float *C, float *B, float *A, int K, int M) {
    return thaBLAS_s_matmul(handle, M, 1, K, A, B, C);
}
__global__ void thaDNN_s_matmulvec_v2_kernel(float *C, float *B, float *A, int K, int M) {
    int gx = blockIdx.x;
    int lx = threadIdx.x;
    float sum = 0.0f;
    for (int k=lx ; k<K ; k+=blockDim.x) {
        sum += A[gx*K + k] * B[k];
    }
    sum = block_reduce_sum(sum);
    if (lx == 0) {
        C[gx] = sum;
    }
}

// A[M,K] x B[K,1] = C[1,M]
thablasStatus_t thaDNN_s_matmulvec_v2(thablasHandle_t* handle, float *C, float *B, float *A, int K, int M) {
    // if (K==0 || M==0 || A == nullptr || B == nullptr || C == nullptr || handle.current_gpu_id < 0)
    // {
    //     printf("THABLAS MAT MUL VEC ERROR: INVALID ARGUMENT\n"); fflush(stdout);
    //     return THABLAS_STATUS_ALLOC_FAILED;        
    // }

    // CHECK_HIP(hipSetDevice(handle.current_gpu_id));
    dim3 blockDim(MAX_BLOCK_SIZE);
    dim3 gridDim(M);

    hipLaunchKernelGGL(thaDNN_s_matmulvec_v2_kernel, gridDim, blockDim, 0, handle->calc_stream, C, B, A, K, M);
    // CHECK_HIP(hipGetLastError());

    return THABLAS_STATUS_SUCCESS;
}


// [B and C are col major
__global__ void thaBLAS_s_matmul_reduction_kernel(float *A, float *B, float *C, int M, int N, int K)
{
    int i = blockIdx.x;
    int j = blockIdx.y;
    int lx = threadIdx.x;
    float sum = 0.0f;

    float *Ccol = C + j * M;
    float *Bcol = B + j * K;
    for (int k=lx ; k<K ; k+=blockDim.x)
    {
        sum += A[i*K + k] * Bcol[k];
    }
    sum = block_reduce_sum(sum);
    if (lx == 0)
    {
        Ccol[i] = sum;
    }
}


// B and C are col major
thablasStatus_t thaBLAS_s_matmul_reduction(thablasHandle_t* handle, float *A, float *B, float *C, int M, int N, int K)
{
    // if (K + M + n_batches==0 || A == nullptr || B_batch == nullptr || C_batch == nullptr || handle.current_gpu_id < 0)
    // {
    //     printf("THABLAS MAT MUL VEC BATCH ERROR: INVALID ARGUMENT\n"); fflush(stdout);
    //     return THABLAS_STATUS_ALLOC_FAILED;        
    // }

    // CHECK_HIP(hipSetDevice(handle.current_gpu_id));
    dim3 blockDim(MAX_BLOCK_SIZE);
    dim3 gridDim(M, N);

    hipLaunchKernelGGL(thaBLAS_s_matmul_reduction_kernel, gridDim, blockDim, 0, handle->calc_stream, A, B, C, M, N ,K);
    // CHECK_HIP(hipGetLastError());

    return THABLAS_STATUS_SUCCESS;
}

__global__ void thaBLAS_s_sgemm_Mx16xK_kernel(const float *A, const float *B, float *D, int M, int N, int K)
{
    using float4 = __attribute__( (__vector_size__(4 * sizeof(float)) )) float;
    float4 dmn = {0};
    int i_off = blockIdx.x * 16;

    for(int t=0 ; t<K ; t+=4)
    {
        int mk = threadIdx.y + t + K * threadIdx.x + i_off * K; // A[x][y] = i * K + k
        // int kn = threadIdx.x + N * threadIdx.y; // B[y][x] = j + k * N -> row major
        int kn = threadIdx.x * K + threadIdx.y + t ; // B[y][x] = j * K + k -> column major

        float amk = A[mk];
        float bkn = B[kn];
        dmn = __builtin_amdgcn_mfma_f32_16x16x4f32(amk, bkn, dmn, 0, 0, 0);

        for (int i = 0; i < 4; ++i) 
        {
            const int idx = threadIdx.x * M + i + threadIdx.y * 4 + i_off;// -> column major
            D[idx] = dmn[i];
        }
    }
}

thablasStatus_t thaBLAS_s_sgemm_Mx16xK(thablasHandle_t* handle, float *d_A, float *d_B, float *d_D, int M, int N, int K)
{
    dim3 blockDim(16, 4, 1);
    dim3 gridDim(M / 16, 1, 1);

    hipLaunchKernelGGL(thaBLAS_s_sgemm_Mx16xK_kernel, gridDim, blockDim, 0, handle->calc_stream, d_A, d_B, d_D, M, N, K);

    return THABLAS_STATUS_SUCCESS;
}

thablasStatus_t thaBLAS_s_matmul_ifdef(thablasHandle_t* handle, float *d_A, float *d_B, float *d_D, int M, int N, int K)
{
#ifdef MATRIX_CORE
    return thaBLAS_s_sgemm_Mx16xK(handle, d_A, d_B, d_D, M, N, K);
#else
    return thaBLAS_s_matmul_reduction(handle, d_A, d_B, d_D, M, N, K);
#endif   
}
