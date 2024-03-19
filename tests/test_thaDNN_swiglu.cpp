
#include "hip_helper.hpp"
#include "thaBLAS.hpp"
#include "thaDNN.hpp"
#include "utils.hpp"

#include <alloca.h>
#include <assert.h>
#include <chrono>

// _h2d_ = host to device
// [hb], [hb2] are allocated on Host
// only run on 1 devices
thablasStatus_t thaDNN_h2d_s_swiglu(float *hb, float *hb2, int hidden_dim)
{
    if ( hidden_dim == 0 || hb == nullptr || hb2 == nullptr)
    {
        printf("THABLAS SwiGLU_non_linearity ERROR: INVALID ARGUMENT\n"); fflush(stdout);
        return THABLAS_STATUS_ALLOC_FAILED;        
    }

    int num_devices;
    CHECK_HIP(hipGetDeviceCount(&num_devices));

    if (!num_devices)
    {
        printf("THABLAS SwiGLU_non_linearity ERROR: COULD NOT FIND ANY COMPUTE DEVICE\n"); fflush(stdout);
        return THABLAS_STATUS_ALLOC_FAILED;
    }

    float *hb_d, *hb2_d;

    CHECK_HIP(hipSetDevice(0));
    CHECK_HIP(hipMalloc(&hb_d, hidden_dim*sizeof(float)));
    CHECK_HIP(hipMalloc(&hb2_d, hidden_dim*sizeof(float)));

    CHECK_HIP(hipMemcpy(hb_d, hb, hidden_dim*sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(hb2_d, hb2, hidden_dim*sizeof(float), hipMemcpyHostToDevice));

    thablasHandle_t handle;
    thablasCreate(&handle);
    thablasStatus_t status = thaDNN_s_swiglu(handle, hb_d, hb2_d, hidden_dim);
    if (status != THABLAS_STATUS_SUCCESS) {
        printf("THABLAS SwiGLU_non_linearity ERROR: ERROR on Device\n"); fflush(stdout);
    }

    CHECK_HIP(hipMemcpy(hb, hb_d, hidden_dim*sizeof(float), hipMemcpyDeviceToHost));
    CHECK_HIP(hipMemcpy(hb2, hb2_d, hidden_dim*sizeof(float), hipMemcpyDeviceToHost));

    CHECK_HIP(hipDeviceSynchronize());
    
    CHECK_HIP(hipFree(hb_d));
    CHECK_HIP(hipFree(hb2_d));

    return THABLAS_STATUS_SUCCESS;
    
}


int main() {
  // TODO: Unitetst operator function
  
  return 0;
}
