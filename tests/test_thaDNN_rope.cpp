#include "hip_helper.hpp"
#include "thaBLAS.hpp"
#include "thaDNN.hpp"
#include "utils.hpp"

#include <alloca.h>
#include <assert.h>
#include <chrono>

// _h2d_ = host to device
// [q], [k] are allocated on Host
// only run on 1 devices
// [dim] % 2 = 0
thablasStatus_t thaDNN_h2d_s_rope(int dim, int head_size, int kv_dim, int pos, float *q, float *k) 
{
    if (dim==0 || head_size==0 || kv_dim==0 || q == nullptr || k == nullptr)
    {
        printf("THABLAS RoPE_relative_positional_encoding ERROR: INVALID ARGUMENT\n"); fflush(stdout);
        return THABLAS_STATUS_ALLOC_FAILED;        
    }

    int num_devices;
    CHECK_HIP(hipGetDeviceCount(&num_devices));

    if (!num_devices)
    {
        printf("THABLAS RoPE_relative_positional_encoding ERROR: COULD NOT FIND ANY COMPUTE DEVICE\n"); fflush(stdout);
        return THABLAS_STATUS_ALLOC_FAILED;
    }

    float *q_d, *k_d;

    CHECK_HIP(hipSetDevice(0));
    CHECK_HIP(hipMalloc(&q_d, dim * sizeof(float)));
    CHECK_HIP(hipMalloc(&k_d, dim * sizeof(float)));

    CHECK_HIP(hipMemcpy(q_d, q, dim * sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(k_d, k, dim * sizeof(float), hipMemcpyHostToDevice));

    thablasHandle_t handle;
    thablasCreate(&handle);
    thablasStatus_t status = thaDNN_s_rope(handle, dim, head_size, kv_dim, pos, q_d, k_d);
    if (status != THABLAS_STATUS_SUCCESS) {
        printf("THABLAS RoPE_relative_positional_encoding ERROR: ERROR on Device\n"); fflush(stdout);
    }

    CHECK_HIP(hipMemcpy(q, q_d, dim * sizeof(float), hipMemcpyDeviceToHost));
    CHECK_HIP(hipMemcpy(k, k_d, dim * sizeof(float), hipMemcpyDeviceToHost));

    CHECK_HIP(hipDeviceSynchronize());

    CHECK_HIP(hipFree(q_d));
    CHECK_HIP(hipFree(k_d));

    return THABLAS_STATUS_SUCCESS;
}

int main() {
  // TODO: Unitetst operator function
  
  return 0;
}
