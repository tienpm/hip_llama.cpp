#include "hip_helper.hpp"
#include "thaBLAS.hpp"
#include "utils.hpp"

#include <alloca.h>
#include <assert.h>
#include <chrono>


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

    for(int gid = 0 ; gid < num_gpus ; ++gid)
    {
        CHECK_HIP(hipSetDevice(gid));
        CHECK_HIP(hipFree(A_gpu[gid]));
        CHECK_HIP(hipFree(B_gpu[gid]));
    }

    return THABLAS_STATUS_SUCCESS;
}

int main() {
  // TODO: Unitetst operator function
  
  return 0;
}
