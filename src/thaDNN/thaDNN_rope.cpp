#include "thaDNN/thaDNN_rmsnorm.hpp"

#define ROUND_UP(N, S)((N + S - 1)/ S)

/*
    RoPE relative positional encoding: complex-valued rotate q and k in each head
    for (int i = 0; i < dim; i+=2) {
      int head_dim = i % head_size;
      float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
      float val = pos * freq;
      float fcr = cosf(val);
      float fci = sinf(val);
      int rotn = i < kv_dim ? 2 : 1; // how many vectors? 2 = q & k, 1 = q only
      for (int v = 0; v < rotn; v++) {
        float* vec = v == 0 ? s->q : s->k; // the vector to rotate (query or key)
        float v0 = vec[i];
        float v1 = vec[i+1];
        vec[i]   = v0 * fcr - v1 * fci;
        vec[i+1] = v0 * fci + v1 * fcr;
      }
    }
*/

__global__ void thaDNN_s_rope_kernel(int dim, int head_size, int kv_dim, int pos, float *q, float *k) {
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    if (i >= dim) return;
    //
    int head_dim = i % head_size;
    float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
    float val = pos * freq;
    float fcr = cosf(val);
    float fci = sinf(val);
    int rotn = i < kv_dim ? 2 : 1; // how many vectors? 2 = q & k, 1 = q only
    for (int v = 0; v < rotn; v++) {
        float* vec = v == 0 ? q : k; // the vector to rotate (query or key)
        float v0 = vec[i];
        float v1 = vec[i+1];
        vec[i]   = v0 * fcr - v1 * fci;
        vec[i+1] = v0 * fci + v1 * fcr;
    }
}

// _s_ = single persion (float)
// input: q, k allocated on device
// [dim] % 2 = 0
thablasStatus_t thaDNN_s_rope(thablasHandle_t* handle, int dim, int head_size, int kv_dim, int pos, float *q, float *k) {
    // if (dim==0 || head_size==0 || kv_dim==0 || q == nullptr || k == nullptr || handle.current_gpu_id < 0)
    // {
    //     fprintf(stderr, "THABLAS RoPE_relative_positional_encoding ERROR: INVALID ARGUMENT\n");
    //     return THABLAS_STATUS_ALLOC_FAILED;        
    // }

    // CHECK_HIP(hipSetDevice(handle.current_gpu_id));
    dim3 blockDim(64);
    dim3 gridDim(ROUND_UP(dim, 128));
    hipLaunchKernelGGL(thaDNN_s_rope_kernel, gridDim, blockDim, 0, handle->calc_stream, dim, head_size, kv_dim, pos, q, k);
    // CHECK_HIP(hipGetLastError());

    return THABLAS_STATUS_SUCCESS;
}
