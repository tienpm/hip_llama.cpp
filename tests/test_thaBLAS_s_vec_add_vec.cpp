#include "hip_helper.hpp"
#include "thaBLAS.hpp"
#include "utils.hpp"

#include <alloca.h>
#include <assert.h>
#include <chrono>

// a[i] += b[i]
thablasStatus_t thaBLAS_h2d_s_vecaddvec(float *a, float *b, int size)
{
    if (a==nullptr || b==nullptr || size==0)
    {
        printf("THABLAS VEC ADD VEC ERROR: INVALID ARGUMENT\n"); fflush(stdout);
        return THABLAS_STATUS_ALLOC_FAILED;        
    }

    int num_gpus;
    CHECK_HIP(hipGetDeviceCount(&num_gpus));

    if (!num_gpus)
    {
        printf("THABLAS VEC ADD VEC ERROR: COULD NOT FIND ANY GPU\n"); fflush(stdout);
        return THABLAS_STATUS_ALLOC_FAILED;
    }

    float *a_d, *b_d;
    CHECK_HIP(hipMalloc(&a_d, size * sizeof(float)));
    CHECK_HIP(hipMalloc(&b_d, size * sizeof(float)));

    CHECK_HIP(hipMemcpy(a_d, a, size * sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(b_d, b, size * sizeof(float), hipMemcpyHostToDevice));

    thablasHandle_t handle;
    thablasCreate(&handle);
    thablasStatus_t status = thaBLAS_s_vecaddvec(handle, a_d, b_d, size);
    if (status != THABLAS_STATUS_SUCCESS) {
        printf("THABLAS VEC ADD VEC ERROR: ERROR\n"); fflush(stdout);
    }

    CHECK_HIP(hipMemcpy(a, a_d, size * sizeof(float), hipMemcpyDeviceToHost));

    CHECK_HIP(hipDeviceSynchronize());

    CHECK_HIP(hipFree(a_d));
    CHECK_HIP(hipFree(b_d));

    return THABLAS_STATUS_SUCCESS;
}

int main() {
  // TODO: Unitetst operator function
  
  return 0;
}
