#include "hip_helper.hpp"
#include "thaBLAS.hpp"
#include "thaDNN.hpp"
#include "utils.hpp"

#include <alloca.h>
#include <assert.h>
#include <chrono>

// // '_h2d_ = host to device
// // o, x, weight allocated on Host
// // only run on 1 devices
// thablasStatus_t thaDNN_h2d_s_rmsnorm_v2(float* o, float* x, float* weight, int size) 
// {
//     if (size==0 || o == nullptr || x == nullptr || weight == nullptr)
//     {
//         printf("THABLAS RMSNORM ERROR: INVALID ARGUMENT\n"); fflush(stdout);
//         return THABLAS_STATUS_ALLOC_FAILED;        
//     }
//
//     int num_devices;
//     CHECK_HIP(hipGetDeviceCount(&num_devices));
//
//     if (!num_devices)
//     {
//         printf("THABLAS RMSNORM ERROR: COULD NOT FIND ANY COMPUTE DEVICE\n"); fflush(stdout);
//         return THABLAS_STATUS_ALLOC_FAILED;
//     }
//
//     float *o_d, *x_d, *weight_d;
//
//     CHECK_HIP(hipSetDevice(0));
//     CHECK_HIP(hipMalloc(&o_d, size * sizeof(float)));
//     CHECK_HIP(hipMalloc(&x_d, size * sizeof(float)));
//     CHECK_HIP(hipMalloc(&weight_d, size * sizeof(float)));
//
//     CHECK_HIP(hipMemcpy(x_d, x, size * sizeof(float), hipMemcpyHostToDevice));
//     CHECK_HIP(hipMemcpy(weight_d, weight, size * sizeof(float), hipMemcpyHostToDevice));
//
//     thablasHandle_t handle;
//     thablasCreate(&handle);
//     thablasStatus_t status = thaDNN_s_rmsnorm_v2(handle, o_d, x_d, weight_d, size);
//     if (status != THABLAS_STATUS_SUCCESS) 
//     {
//         printf("THABLAS RMSNORM ERROR: ERROR on Device\n"); fflush(stdout);
//     }
//
//     CHECK_HIP(hipMemcpy(o, o_d, size * sizeof(float), hipMemcpyDeviceToHost));
//
//     CHECK_HIP(hipDeviceSynchronize());
//
//     CHECK_HIP(hipFree(o_d));
//     CHECK_HIP(hipFree(x_d));
//     CHECK_HIP(hipFree(weight_d));
//
//     return THABLAS_STATUS_SUCCESS;
// }

int main() {
  // TODO: Unitetst operator function
  
  return 0;
}

