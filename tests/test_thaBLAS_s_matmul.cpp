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

bool check_mat_mul(float *A, float *B, float *C, int M, int N, int K) {
    float *C_ans;
    alloc_mat(&C_ans, M, N);
    zero_mat(C_ans, M, N);

    #pragma omp parallel for num_threads(20)
    for (int i = 0; i < M; ++i) {
        for (int k = 0; k < K; ++k) {
        for (int j = 0; j < N; ++j) {
            C_ans[i * N + j] += A[i * K + k] * B[k * N + j];
        }
        }
    }

    bool is_valid = true;
    int cnt = 0, thr = 10;
    float eps = 1e-3;
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
        float c = C[i * N + j];
        float c_ans = C_ans[i * N + j];
        if (fabsf(c - c_ans) > eps &&
            (c_ans == 0 || fabsf((c - c_ans) / c_ans) > eps)) {
            ++cnt;
            if (cnt <= thr)
            printf("C[%d][%d] : correct_value = %f, your_value = %f\n", i, j, c_ans, c);
            if (cnt == thr + 1)
            printf("Too many error, only first %d values are printed.\n", thr);
            is_valid = false;
        }
        }
    }

    // for (int i = 0; i < M; ++i) {
    //   for (int j = 0; j < K; ++j) 
    //     printf("%0.2f ", A[i * K + j]);
    //   printf("\n");
    // }
    // printf("\n");
    // for (int i = 0; i < K; ++i) {
    //   for (int j = 0; j < N; ++j) 
    //     printf("%0.2f ", B[i * N + j]);
    //   printf("\n");
    // }
    // printf("\n"); fflush(stdout);

    util_free((void*)A);
    util_free((void*)B);
    util_free((void*)C);
    util_free((void*)C_ans);

    if (is_valid) {
        printf("Validation: VALID\n");  fflush(stdout);
        return 1;
    } else {
        printf("Validation: INVALID\n");  fflush(stdout);
        return 0;
    }
}

bool thaBLAS_h2d_s_sgemm_16x16x4()
{
    int M = 16;
    int N = 16;
    int K = 4;
    float *A, *B, *C;
    alloc_mat(&A, M, K);
    alloc_mat(&B, K, N);
    alloc_mat(&C, M, N);
    rand_mat(A, M, K);
    rand_mat(B, K, N);
    zero_mat(C, M, N);

    float *d_A;
    float *d_B;
    float *d_C;
    CHECK_HIP(hipMalloc(&d_A, M * K * sizeof(float)));
    CHECK_HIP(hipMalloc(&d_B, K * N * sizeof(float)));
    CHECK_HIP(hipMalloc(&d_C, M * N * sizeof(float)));

    CHECK_HIP(hipMemcpy(d_A, A, M * K * sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(d_B, B, K * N * sizeof(float), hipMemcpyHostToDevice));

    thablasHandle_t handle;
    thablasCreate(&handle);
    thaBLAS_s_sgemm_16x16x4(handle, d_A, d_B, d_C, M, N, K);

    CHECK_HIP(hipMemcpy(C, d_C, M * N * sizeof(float), hipMemcpyDeviceToHost));
    CHECK_HIP(hipDeviceSynchronize());
        
    return check_mat_mul(A, B, C, M, N, K);
}

int main() {
  // TODO: Unitetst operator function
  bool valid = 1;

  valid = thaBLAS_h2d_s_sgemm_16x16x4();
  assert(valid);

  return 0;
}
