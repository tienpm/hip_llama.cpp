#include "thaDNN/thaDNN_swiglu.hpp"

// '_s_' = single persion (float)
__global__ void thaDNN_s_swiglu_kernel(float* hb, float*hb2, int hidden_dim) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= hidden_dim) return;
      // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
    float val = hb[i];
    val *= (1.0f / (1.0f + expf(-val)));
      // elementwise multiply with w3(x)
    val *= hb2[i];
    hb[i] = val;
}


// '_s_' = single prisesion
// input: hb, hb2 allocated on device
thablasStatus_t thaDNN_s_swiglu(thablasHandle_t* handle, float *hb, float *hb2, int hidden_dim) {
    // if (hidden_dim==0 || hb == nullptr || hb2 == nullptr || handle.current_gpu_id < 0)
    // {
    //     printf("THABLAS SwiGLU_non_linearity ERROR: INVALID ARGUMENT\n"); fflush(stdout);
    //     return THABLAS_STATUS_ALLOC_FAILED;        
    // }

    // CHECK_HIP(hipSetDevice(handle.current_gpu_id));
    dim3 blockDim(64);
    dim3 gridDim((hidden_dim + blockDim.x - 1) / blockDim.x);
    hipLaunchKernelGGL(thaDNN_s_swiglu_kernel, 
                       gridDim, blockDim, 0, handle->calc_stream, 
                       hb, hb2, hidden_dim);
    // CHECK_HIP(hipGetLastError());

    return THABLAS_STATUS_SUCCESS;
}
