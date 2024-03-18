#include "thaDNN.hpp"
#include "thaBLAS.hpp"
#include "hip_helper.hpp"
#include "seq.hpp"

#include <hip/hip_runtime.h>
#include <omp.h>
#include <hipblas.h>

#define WARP_SIZE 64
// size = 1 -> 16384
#define RMS_LOCAL_BANK_SIZE 8

const int TILE_SIZE = 4; // batch >= TILE_SIZE*VECTOR_SIZE
const int VECTOR_SIZE = 4;  // TILE_SIZE >= VECTOR_SIZE

__global__ void thaDNN_s_rmsnorm_kernel(float* o, float* x, float* weight, int size)
{
    int j = threadIdx.x;
    int j_pos = j * 16;
    if (j_pos >= size) return;
    // 768  = 16 * 16 * 3
    // 4096 = 16 * 16 * 16
    // 5120 = 16 * 16 * 20
    // 8192 = 16 * 16 * 32 
    extern __shared__ float local_s[];
    float local_x[16];
    float ss = 0.0f;
    for(int i=0 ; i<16 ; ++i)
    {
        if (j_pos+i < size)
        {
            local_x[i] = x[j_pos+i];
            ss += local_x[i] * local_x[i];
        }
    }
    local_s[j * RMS_LOCAL_BANK_SIZE] = ss;
    __syncthreads();
    
    ss = 0.0f;
    if (j%16==0)
    {
        int i_last = std::min((int)blockDim.x, j+16);
        for(int i=j ; i<i_last; ++i)
            ss += local_s[i * RMS_LOCAL_BANK_SIZE];
        local_s[j * RMS_LOCAL_BANK_SIZE] = ss;
        // printf("j: %d\n", j);
    }
    __syncthreads();

    ss = 0;
    if (j==0)
    {
        int i_last = blockDim.x;
        for(int i=0 ; i<i_last; i+=16)
        {
            ss += local_s[i * RMS_LOCAL_BANK_SIZE];
            // printf("ss i: %f %d\n", ss, i);
        }
        ss /= size;
        ss += 1e-5f;
        ss = 1.0f / sqrtf(ss);
        local_s[0] = ss;
    }
    __syncthreads();

    ss = local_s[0];
    for(int i=0 ; i<16 ; ++i)
        if (j_pos + i < size)
        {
            o[j_pos + i] = weight[j_pos + i] * (ss * local_x[i]);
        }
}

// '_s_' = single persion (float)
// input: o, x, weight allocated on device
// input: size = 1 -> 16384
thablasStatus_t thaDNN_s_rmsnorm(thablasHandle_t handle, float* o, float* x, float* weight, int size) 
{
    if (size==0 || o == nullptr || x == nullptr || weight == nullptr || handle.current_gpu_id < 0)
    {
        printf("THABLAS RMSNORM ERROR: INVALID ARGUMENT\n"); fflush(stdout);
        return THABLAS_STATUS_ALLOC_FAILED;        
    }

    CHECK_HIP(hipSetDevice(handle.current_gpu_id));
    int num_device_threads = (size + 16 - 1) / 16;
    dim3 blockDim(num_device_threads);
    dim3 gridDim(1);
    int local_mem_size = (num_device_threads * RMS_LOCAL_BANK_SIZE) * sizeof(float);
    hipLaunchKernelGGL(thaDNN_s_rmsnorm_kernel, gridDim, blockDim, local_mem_size, 0, o, x, weight, size);
    CHECK_HIP(hipGetLastError());

    return THABLAS_STATUS_SUCCESS;
}

// _h2d_ = host to device
// o, x, weight allocated on Host
// only run on 1 devices
thablasStatus_t thaDNN_h2d_s_rmsnorm(float* o, float* x, float* weight, int size) 
{
    if (size==0 || o == nullptr || x == nullptr || weight == nullptr)
    {
        printf("THABLAS RMSNORM ERROR: INVALID ARGUMENT\n"); fflush(stdout);
        return THABLAS_STATUS_ALLOC_FAILED;        
    }

    int num_devices;
    CHECK_HIP(hipGetDeviceCount(&num_devices));

    if (!num_devices)
    {
        printf("THABLAS RMSNORM ERROR: COULD NOT FIND ANY COMPUTE DEVICE\n"); fflush(stdout);
        return THABLAS_STATUS_ALLOC_FAILED;
    }

    float *o_d, *x_d, *weight_d;

    CHECK_HIP(hipSetDevice(0));
    CHECK_HIP(hipMalloc(&o_d, size * sizeof(float)));
    CHECK_HIP(hipMalloc(&x_d, size * sizeof(float)));
    CHECK_HIP(hipMalloc(&weight_d, size * sizeof(float)));

    CHECK_HIP(hipMemcpy(x_d, x, size * sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(weight_d, weight, size * sizeof(float), hipMemcpyHostToDevice));

    thablasHandle_t handle;
    thablasCreate(&handle);
    thablasStatus_t status = thaDNN_s_rmsnorm(handle, o_d, x_d, weight_d, size);
    if (status != THABLAS_STATUS_SUCCESS) {
        printf("THABLAS RMSNORM ERROR: ERROR on Device\n"); fflush(stdout);
    }

    CHECK_HIP(hipMemcpy(o, o_d, size * sizeof(float), hipMemcpyDeviceToHost));

    CHECK_HIP(hipDeviceSynchronize());

    CHECK_HIP(hipFree(o_d));
    CHECK_HIP(hipFree(x_d));
    CHECK_HIP(hipFree(weight_d));

    return THABLAS_STATUS_SUCCESS;
}

/*
*********************************************************************************************************
* softmax
*********************************************************************************************************
*/

// size = 1 -> 32000
__global__ void thaDNN_s_softmax_kernel(float* x, int size)
{
    /*
    reduction for max value 
    */

    // split array to blocks ,each block is responsible for 32 elements
    int j = threadIdx.x;
    int j_pos = j * 32;
    if (j_pos >= size) return;

    
    // find max value (for numerical stability)
    __shared__ float local_reduction_max[1000]; // 32000/32 = 1000
    float max_val = x[j_pos];
    float local_x[32];

    #pragma unroll
    for (int i = 0; i < 32; i++)
    {
        if (j_pos + i < size)
        {
            local_x[i] = x[j_pos + i];
            if (local_x[i] > max_val) max_val = local_x[i];
        }
    }

    local_reduction_max[j] = max_val;
    __syncthreads();


    // reduction for max value with step 32
    if (j_pos==0)
    {
        max_val = local_reduction_max[0];
        int i_stop = std::min(1000, (int)blockDim.x);
        #pragma unroll
        for (int i = 0; i < i_stop; i++)
        {
            if (local_reduction_max[i] > max_val) max_val = local_reduction_max[i];
        }
        local_reduction_max[0] = max_val;
    }

    __syncthreads();

    /*
    exp and sum
    */

    
    // split array to blocks ,each block is responsible for 32 elements
    __shared__ float local_reduction_sum[1000]; // 32000/32 = 1000
    float sum = 0.0f;
    max_val = local_reduction_max[0];
    #pragma unroll
    for (int i = 0; i < 32; i++)
    {
        if (j_pos + i < size) 
        {
            local_x[i] = expf(local_x[i] - max_val);
            sum += local_x[i];
        }
    }

    //  reduction for sum with step 32
    local_reduction_sum[j] = sum;
    __syncthreads();

    if (j_pos==0)
    {
        sum = 0.0f;
        int i_stop = std::min(1000, (int)blockDim.x);
        #pragma unroll
        for (int i = 0; i < i_stop; i++)
        {
            sum += local_reduction_sum[i];
        }
        local_reduction_sum[0] = sum;
    }

    __syncthreads();

    /*
    normalize
    */
    sum = local_reduction_sum[0];
    #pragma unroll
    for (int i = 0; i < 32; i++)
    {
        if (j_pos + i < size) 
            x[j_pos + i] = local_x[i] / sum;
    }

}

// _s_ = single persion (float)
// input: output, x allocated on device
// input: size = 32000
thablasStatus_t thaDNN_s_softmax(thablasHandle_t handle, float* x, int size) 
{
    if (size==0 || x == nullptr || handle.current_gpu_id < 0)
    {
        printf("THABLAS SOFTMAX ERROR: INVALID ARGUMENT\n"); fflush(stdout);
        return THABLAS_STATUS_ALLOC_FAILED;        
    }

    CHECK_HIP(hipSetDevice(handle.current_gpu_id));
    dim3 blockDim((size + 32 - 1) / 32);
    dim3 gridDim(1);
    // dim3 gridSize((size + SOFTMAX_BLOCK_SIZE - 1) / SOFTMAX_BLOCK_SIZE);
    hipLaunchKernelGGL(thaDNN_s_softmax_kernel, gridDim, blockDim, 0, 0, x, size);
    CHECK_HIP(hipGetLastError());

    return THABLAS_STATUS_SUCCESS;
}

// _h2d_ = host to device
// [output], [x] are allocated on Host
// only run on 1 devices
// [size] = 1 -> 32000
thablasStatus_t thaDNN_h2d_s_softmax(float* x, int size) 
{
    if (size==0 || x == nullptr)
    {
        printf("THABLAS SOFTMAX ERROR: INVALID ARGUMENT\n"); fflush(stdout);
        return THABLAS_STATUS_ALLOC_FAILED;        
    }

    int num_devices;
    CHECK_HIP(hipGetDeviceCount(&num_devices));

    if (!num_devices)
    {
        printf("THABLAS SOFTMAX ERROR: COULD NOT FIND ANY COMPUTE DEVICE\n"); fflush(stdout);
        return THABLAS_STATUS_ALLOC_FAILED;
    }

    float *x_d;

    CHECK_HIP(hipSetDevice(0));
    CHECK_HIP(hipMalloc(&x_d, size * sizeof(float)));

    CHECK_HIP(hipMemcpy(x_d, x, size * sizeof(float), hipMemcpyHostToDevice));

    thablasHandle_t handle;
    thablasCreate(&handle);
    thablasStatus_t status = thaDNN_s_softmax(handle, x_d, size);
    if (status != THABLAS_STATUS_SUCCESS) {
        printf("THABLAS SOFTMAX ERROR: ERROR on Device\n"); fflush(stdout);
    }

    CHECK_HIP(hipMemcpy(x, x_d, size * sizeof(float), hipMemcpyDeviceToHost));

    CHECK_HIP(hipDeviceSynchronize());

    CHECK_HIP(hipFree(x_d));

    return THABLAS_STATUS_SUCCESS;
}


/*
*********************************************************************************************************
*  RoPE relative positional encoding
*********************************************************************************************************
*/

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

__global__ void thaDNN_s_rope_kernel(int dim, int head_size, int kv_dim, int pos, float *q, float *k)
{
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
thablasStatus_t thaDNN_s_rope(thablasHandle_t handle, int dim, int head_size, int kv_dim, int pos, float *q, float *k) 
{
    if (dim==0 || head_size==0 || kv_dim==0 || q == nullptr || k == nullptr || handle.current_gpu_id < 0)
    {
        printf("THABLAS RoPE_relative_positional_encoding ERROR: INVALID ARGUMENT\n"); fflush(stdout);
        return THABLAS_STATUS_ALLOC_FAILED;        
    }

    CHECK_HIP(hipSetDevice(handle.current_gpu_id));
    dim3 blockDim(64);
    dim3 gridDim((dim + 128 - 1) / 128);
    hipLaunchKernelGGL(thaDNN_s_rope_kernel, gridDim, blockDim, 0, 0, dim, head_size, kv_dim, pos, q, k);
    CHECK_HIP(hipGetLastError());

    return THABLAS_STATUS_SUCCESS;
}

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


/*
*********************************************************************************************************
*  SwiGLU non-linearity
*********************************************************************************************************
*/

// '_s_' = single persion (float)
__global__ void thaDNN_s_swiglu_kernel(float* hb, float*hb2, int hidden_dim){
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
thablasStatus_t thaDNN_s_swiglu(thablasHandle_t handle, float *hb, float *hb2, int hidden_dim)
{
    if (hidden_dim==0 || hb == nullptr || hb2 == nullptr || handle.current_gpu_id < 0)
    {
        printf("THABLAS SwiGLU_non_linearity ERROR: INVALID ARGUMENT\n"); fflush(stdout);
        return THABLAS_STATUS_ALLOC_FAILED;        
    }

    CHECK_HIP(hipSetDevice(handle.current_gpu_id));
    dim3 blockDim(64);
    dim3 gridDim((hidden_dim + blockDim.x - 1) / blockDim.x);
    hipLaunchKernelGGL(thaDNN_s_swiglu_kernel, gridDim, blockDim, 0, 0, hb, hb2, hidden_dim);
    CHECK_HIP(hipGetLastError());

    return THABLAS_STATUS_SUCCESS;
}

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


/*
*********************************************************************************************************
* Multi-heads
*********************************************************************************************************
*/

__global__ void thaDNN_s_multiheads_1_kernel(int pos, int n_heads, float *s_q, float *s_att, float *s_key_cache, int head_size, int p_seq_len, int loff, int kv_dim, int kv_mul)
{
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  int h = blockIdx.y * blockDim.y + threadIdx.y;
  if (t>pos || h>=n_heads) return;

  float* q = s_q + h * head_size;
  float* att = s_att + h * p_seq_len;

  float* k = s_key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
  float score = 0.0f;
  for (int i = 0; i < head_size; i++) {
    score += q[i] * k[i];
  }
  score /= sqrtf(head_size);
  // save the score to the attention buffer
  att[t] = score;
}

thablasStatus_t thaDNN_s_multiheads_1(thablasHandle_t handle, int pos, int n_heads, float *s_q, float *s_att, float *s_key_cache, int head_size, int seq_len, int loff, int kv_dim, int kv_mul)
{
  if (s_q==nullptr || s_att==nullptr || s_key_cache==nullptr || head_size==0 || seq_len==0 || kv_dim==0)
  {
      printf("THABLAS MULTI_HEADS_1 ERROR: INVALID ARGUMENT\n"); fflush(stdout);
      return THABLAS_STATUS_ALLOC_FAILED;        
  }

  CHECK_HIP(hipSetDevice(handle.current_gpu_id));
  dim3 blockDim(16, 4);
  dim3 gridDim((pos+1 + 16 - 1) / 16, (n_heads + 4 - 1) / 4);
  // CAUTION: careful playing with [pos]. 
  hipLaunchKernelGGL(thaDNN_s_multiheads_1_kernel, gridDim, blockDim, 0, 0, pos, n_heads, s_q, s_att, s_key_cache, head_size, seq_len, loff, kv_dim, kv_mul);
  CHECK_HIP(hipGetLastError());

  return THABLAS_STATUS_SUCCESS;
}

thablasStatus_t thaDNN_h2d_s_multiheads_1(Config* p, RunState* s, int head_size, int pos, int loff, int kv_dim, int kv_mul)
{
    if (p==nullptr || s==nullptr || head_size==0 || kv_dim==0)
    {
        printf("THABLAS MULTI_HEADS_1 ERROR: INVALID ARGUMENT\n"); fflush(stdout);
        return THABLAS_STATUS_ALLOC_FAILED;        
    }

    int num_devices;
    CHECK_HIP(hipGetDeviceCount(&num_devices));

    if (!num_devices)
    {
        printf("THABLAS MULTI_HEADS_1 ERROR: COULD NOT FIND ANY COMPUTE DEVICE\n"); fflush(stdout);
        return THABLAS_STATUS_ALLOC_FAILED;
    }

    int dim = p->dim;
    int n_heads = p->n_heads;
    int seq_len = p->seq_len;
    int n_layers = p->n_layers;
    float *s_q_d;
    float *s_att_d;
    float *s_key_cache_d;
    CHECK_HIP(hipMalloc(&s_q_d, dim * sizeof(float)));
    CHECK_HIP(hipMalloc(&s_att_d, n_heads * seq_len * sizeof(float)));
    CHECK_HIP(hipMalloc(&s_key_cache_d, n_layers * seq_len * kv_dim * sizeof(float)));

    CHECK_HIP(hipMemcpy(s_q_d, s->q, dim * sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(s_key_cache_d, s->key_cache, n_layers * seq_len * kv_dim * sizeof(float), hipMemcpyHostToDevice));

    thablasHandle_t handle;
    thablasCreate(&handle);
    thablasStatus_t status = thaDNN_s_multiheads_1(handle, pos, n_heads, s_q_d, s_att_d, s_key_cache_d, head_size, seq_len, loff, kv_dim, kv_mul);
    if (status != THABLAS_STATUS_SUCCESS) {
        printf("THABLAS MULTI_HEADS_1 ERROR: ERROR on Device\n"); fflush(stdout);
    }

    CHECK_HIP(hipMemcpy(s->att ,s_att_d, n_heads * seq_len * sizeof(float), hipMemcpyDeviceToHost));

    CHECK_HIP(hipDeviceSynchronize());

    CHECK_HIP(hipFree(s_q_d));
    CHECK_HIP(hipFree(s_att_d));
    CHECK_HIP(hipFree(s_key_cache_d));

    return THABLAS_STATUS_SUCCESS;
}

__global__ void thaDNN_s_multiheads_3_kernel(int pos, int n_heads, float *s_xb, float *s_att, float *s_value_cache, int head_size, int seq_len, int loff, int kv_dim, int kv_mul)
{
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  int h = blockIdx.y * blockDim.y + threadIdx.y;
  if (t>=pos+1 || h>=n_heads) return;

  float* xb = s_xb + h * head_size;
  float* att = s_att + h * seq_len;

  float* v = s_value_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
  float a = att[t];

  for (int i = 0; i < head_size; i++) {
    // xb[i] += a * v[i];
    atomicAdd(&xb[i], a*v[i]);
  }
}

thablasStatus_t thaDNN_s_multiheads_3(thablasHandle_t handle, int pos, int n_heads, float *s_xb, float *s_att, float *s_value_cache, int head_size, int seq_len, int loff, int kv_dim, int kv_mul, int dim)
{
  if (s_xb==nullptr || s_att==nullptr || s_value_cache==nullptr || head_size==0 || seq_len==0 || kv_dim==0)
  {
      printf("THABLAS MULTI_HEADS_3 ERROR: INVALID ARGUMENT\n"); fflush(stdout);
      return THABLAS_STATUS_ALLOC_FAILED;        
  }

  CHECK_HIP(hipSetDevice(handle.current_gpu_id));
  CHECK_HIP(hipMemset(s_xb, 0, dim * sizeof(float)));
  dim3 blockDim(16, 4);
  dim3 gridDim((pos+1 + 16 - 1) / 16, (n_heads + 4 - 1) / 4);
  // CAUTION: careful playing with [pos]. 
  hipLaunchKernelGGL(thaDNN_s_multiheads_3_kernel, gridDim, blockDim, 0, 0, pos, n_heads, s_xb, s_att, s_value_cache, head_size, seq_len, loff, kv_dim, kv_mul);
  CHECK_HIP(hipGetLastError());

  return THABLAS_STATUS_SUCCESS;
}

thablasStatus_t thaDNN_h2d_s_multiheads_3(Config* p, RunState* s, int head_size, int pos, int loff, int kv_dim, int kv_mul)
{
  if (p==nullptr || s==nullptr || head_size==0 || kv_dim==0)
  {
      printf("THABLAS MULTI_HEADS_1 ERROR: INVALID ARGUMENT\n"); fflush(stdout);
      return THABLAS_STATUS_ALLOC_FAILED;        
  }

  int num_devices;
  CHECK_HIP(hipGetDeviceCount(&num_devices));

  if (!num_devices)
  {
      printf("THABLAS MULTI_HEADS_1 ERROR: COULD NOT FIND ANY COMPUTE DEVICE\n"); fflush(stdout);
      return THABLAS_STATUS_ALLOC_FAILED;
  }

  int dim = p->dim;
  int n_heads = p->n_heads;
  int seq_len = p->seq_len;
  int n_layers = p->n_layers;
  float *s_xb_d;
  float *s_att_d;
  float *s_value_cache_d;
  CHECK_HIP(hipMalloc(&s_xb_d, dim * sizeof(float)));
  CHECK_HIP(hipMalloc(&s_att_d, n_heads * seq_len * sizeof(float)));
  CHECK_HIP(hipMalloc(&s_value_cache_d, n_layers * seq_len * kv_dim * sizeof(float)));

  CHECK_HIP(hipMemcpy(s_xb_d, s->xb, dim * sizeof(float), hipMemcpyHostToDevice));
  CHECK_HIP(hipMemcpy(s_att_d, s->att, n_heads * seq_len * sizeof(float), hipMemcpyHostToDevice));
  CHECK_HIP(hipMemcpy(s_value_cache_d, s->value_cache, n_layers * seq_len * kv_dim * sizeof(float), hipMemcpyHostToDevice));

  thablasHandle_t handle;
  thablasCreate(&handle);
  thablasStatus_t status = thaDNN_s_multiheads_3(handle, pos, n_heads, s_xb_d, s_att_d, s_value_cache_d, head_size, seq_len, loff, kv_dim, kv_mul, dim);
  if (status != THABLAS_STATUS_SUCCESS) {
      printf("THABLAS MULTI_HEADS_3 ERROR: ERROR on Device\n"); fflush(stdout);
  }

  CHECK_HIP(hipMemcpy(s->xb ,s_xb_d, dim * sizeof(float), hipMemcpyDeviceToHost));

  CHECK_HIP(hipDeviceSynchronize());

  CHECK_HIP(hipFree(s_xb_d));
  CHECK_HIP(hipFree(s_att_d));
  CHECK_HIP(hipFree(s_value_cache_d));

  return THABLAS_STATUS_SUCCESS;
}

thablasStatus_t thaDNN_h2d_s_forward(Transformer* transformer, int token, int pos, float* &output_logits) {

  // a few convenience variables
  Config* p = &transformer->config;
  TransformerWeights* w = &transformer->weights;
  RunState* s = &transformer->state;
  float *x = s->x;
  int dim = p->dim;
  int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
  int kv_mul = p->n_heads / p->n_kv_heads; // integer multiplier of the kv sharing in multiquery
  int hidden_dim =  p->hidden_dim;
  int head_size = dim / p->n_heads;

  thablasStatus_t thablas_status = THABLAS_STATUS_SUCCESS;

  // copy the token embedding into x
  float* content_row = w->token_embedding_table + token * dim;
  memcpy(x, content_row, dim*sizeof(*x));

  // forward all the layers
  for(unsigned long long l = 0; l < p->n_layers; l++) {

    // attention rmsnorm
    // rmsnorm(s->xb, x, w->rms_att_weight + l*dim, dim);
    thablas_status = thaDNN_h2d_s_rmsnorm(s->xb, x, w->rms_att_weight + l*dim, dim);

    // key and value point to the kv cache
    int loff = l * p->seq_len * kv_dim; // kv cache layer offset for convenience
    s->k = s->key_cache + loff + pos * kv_dim;
    s->v = s->value_cache + loff + pos * kv_dim;

    // qkv matmuls for this position
    // matmul(s->q, s->xb, w->wq + l*dim*dim, dim, dim);
    // matmul(s->k, s->xb, w->wk + l*dim*kv_dim, dim, kv_dim);
    // matmul(s->v, s->xb, w->wv + l*dim*kv_dim, dim, kv_dim);
    thablas_status = thaBLAS_h2d_s_matmulvec(s->q, s->xb, w->wq + l*dim*dim, dim, dim);
    thablas_status = thaBLAS_h2d_s_matmulvec(s->k, s->xb, w->wk + l*dim*kv_dim, dim, kv_dim);
    thablas_status = thaBLAS_h2d_s_matmulvec(s->v, s->xb, w->wv + l*dim*kv_dim, dim, kv_dim);

    // RoPE relative positional encoding: complex-valued rotate q and k in each head
    // for (int i = 0; i < dim; i+=2) {
    //   int head_dim = i % head_size;
    //   float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
    //   float val = pos * freq;
    //   float fcr = cosf(val);
    //   float fci = sinf(val);
    //   int rotn = i < kv_dim ? 2 : 1; // how many vectors? 2 = q & k, 1 = q only
    //   for (int v = 0; v < rotn; v++) {
    //     float* vec = v == 0 ? s->q : s->k; // the vector to rotate (query or key)
    //     float v0 = vec[i];
    //     float v1 = vec[i+1];
    //     vec[i]   = v0 * fcr - v1 * fci;
    //     vec[i+1] = v0 * fci + v1 * fcr;
    //   }
    // }
    thablas_status = thaDNN_h2d_s_rope(dim, head_size, kv_dim, pos, s->q, s->k);

    // multihead attention. iterate over all heads
    {
      int h;
      // PART 1
      thablas_status = thaDNN_h2d_s_multiheads_1(p, s, head_size, pos, loff, kv_dim, kv_mul);
      // for (h = 0; h < p->n_heads; h++) {
      //   // get the query vector for this head
      //   float* q = s->q + h * head_size;
      //   // attention scores for this head
      //   float* att = s->att + h * p->seq_len;
      //   // iterate over all timesteps, including the current one
      //   for (int t = 0; t <= pos; t++) {
      //     // get the key vector for this head and at this timestep
      //     float* k = s->key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
      //     // calculate the attention score as the dot product of q and k
      //     float score = 0.0f;
      //     for (int i = 0; i < head_size; i++) {
      //       score += q[i] * k[i];
      //     }
      //     score /= sqrtf(head_size);
      //     // save the score to the attention buffer
      //     att[t] = score;
      //   }
      // }

      // PART 2
      for (h = 0; h < p->n_heads; h++) {
        float* att = s->att + h * p->seq_len;
        // softmax the scores to get attention weights, from 0..pos inclusively
        // softmax(att, pos + 1);
        thablas_status = thaDNN_h2d_s_softmax(att, pos + 1);
      }

      // PART 3
      // memset(s->xb, 0, dim * sizeof(float));
      thablas_status = thaDNN_h2d_s_multiheads_3(p, s, head_size, pos, loff, kv_dim, kv_mul);
      // for (h = 0; h < p->n_heads; h++) {
      //   float* att = s->att + h * p->seq_len;
      //   float* xb = s->xb + h * head_size;
      //   // memset(xb, 0, head_size * sizeof(float));
      //   for (int t = 0; t <= pos; t++) {
      //     // get the value vector for this head and at this timestep
      //     float* v = s->value_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
      //     // get the attention weight for this timestep
      //     float a = att[t];
      //     // accumulate the weighted value into xb
      //     for (int i = 0; i < head_size; i++) {
      //       xb[i] += a * v[i];
      //     }
      //   }
      // }
    // end multi-head
    }

    // final matmul to get the output of the attention
    // matmul(s->xb2, s->xb, w->wo + l*dim*dim, dim, dim);
    thablas_status = thaBLAS_h2d_s_matmulvec(s->xb2, s->xb, w->wo + l*dim*dim, dim, dim);

    // residual connection back into x
    // for (int i = 0; i < dim; i++) {
    //   x[i] += s->xb2[i];
    // }
    thablas_status = thaBLAS_h2d_s_vecaddvec(x, s->xb2, dim);

    // ffn rmsnorm
    // rmsnorm(s->xb, x, w->rms_ffn_weight + l*dim, dim);
    thablas_status = thaDNN_h2d_s_rmsnorm(s->xb, x, w->rms_ffn_weight + l*dim, dim);

    // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
    // first calculate self.w1(x) and self.w3(x)
    // matmul(s->hb, s->xb, w->w1 + l*dim*hidden_dim, dim, hidden_dim);
    // matmul(s->hb2, s->xb, w->w3 + l*dim*hidden_dim, dim, hidden_dim);
    thablas_status = thaBLAS_h2d_s_matmulvec(s->hb, s->xb, w->w1 + l*dim*hidden_dim, dim, hidden_dim);
    thablas_status = thaBLAS_h2d_s_matmulvec(s->hb2, s->xb, w->w3 + l*dim*hidden_dim, dim, hidden_dim);

    // SwiGLU non-linearity
    // for (int i = 0; i < hidden_dim; i++) {
    //   float val = s->hb[i];
    //   // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
    //   val *= (1.0f / (1.0f + expf(-val)));
    //   // elementwise multiply with w3(x)
    //   val *= s->hb2[i];
    //   s->hb[i] = val;
    // }
    thablas_status = thaDNN_h2d_s_swiglu(s->hb, s->hb2, hidden_dim);

    // final matmul to get the output of the ffn
    // matmul(s->xb, s->hb, w->w2 + l*dim*hidden_dim, hidden_dim, dim);
    thablas_status = thaBLAS_h2d_s_matmulvec(s->xb, s->hb, w->w2 + l*dim*hidden_dim, hidden_dim, dim);

    // residual connection
    // for (int i = 0; i < dim; i++) {
    //   x[i] += s->xb[i];
    // }
    thablas_status = thaBLAS_h2d_s_vecaddvec(x, s->xb, dim);
  }

  // final rmsnorm
  // rmsnorm(x, x, w->rms_final_weight, dim);
  thablas_status = thaDNN_h2d_s_rmsnorm(x, x, w->rms_final_weight, dim);

  // classifier into logits
  // matmul(s->logits, x, w->wcls, p->dim, p->vocab_size);
  thablas_status = thaBLAS_h2d_s_matmulvec(s->logits, x, w->wcls, p->dim, p->vocab_size);
  
  output_logits = s->logits;
  return thablas_status;
}

/*
***********************************************************************************************************************************************************************************
* rmsnorm using shuffle and reduce
***********************************************************************************************************************************************************************************
*/

// biến warpSize là biến built-in của HIP, mặc định là 64
__inline__ __device__
float warpReduceSum(float val) {
  // printf("warpSize: %d\n", warpSize);
  for (int offset = warpSize/2; offset > 0; offset /= 2) 
    val += __shfl_down(val, offset);
  return val;
}

__device__ float warp_reduce_sum(float val)
{
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) 
        val += __shfl_xor(val, offset);
    return val;
}

__device__ float block_reduce_sum(float val) 
{
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

// modify deviceReduceBlockAtomicKernel to caculate sum of squares
__global__ void thaDNN_s_rmsnorm_kernel_v2(float* o, float* x, float* weight, int size)
{
    int lx = threadIdx.x;
    // int gx = blockIdx.x;
    float tmp;
    float ss = 0.0;
    __shared__ float total_sum;

    for(int i = lx; i < size; i += blockDim.x) {
        tmp = x[i];
        ss += tmp * tmp;
    }

    ss = block_reduce_sum(ss);
    
    if (lx == 0)
    {
        ss /= size;
        ss += 1e-5f;
        ss = 1.0f / sqrtf(ss);
        total_sum = ss;
    }
    __syncthreads();

    ss = total_sum;
    for(int i = lx; i < size; i += blockDim.x) {
        o[i] = weight[i] * (ss * x[i]);
    }
}

// '_s_' = single persion (float)
// input: o, x, weight allocated on device
// input: size = 1 -> 16384
thablasStatus_t thaDNN_s_rmsnorm_v2(thablasHandle_t handle, float* o, float* x, float* weight, int size) 
{
    if (size==0 || o == nullptr || x == nullptr || weight == nullptr || handle.current_gpu_id < 0)
    {
        printf("THABLAS RMSNORM ERROR: INVALID ARGUMENT\n"); fflush(stdout);
        return THABLAS_STATUS_ALLOC_FAILED;        
    }

    CHECK_HIP(hipSetDevice(handle.current_gpu_id));
    dim3 blockDim(1024);
    dim3 gridDim(1);
    hipLaunchKernelGGL(thaDNN_s_rmsnorm_kernel_v2, gridDim, blockDim, 0, 0, o, x, weight, size);
    
    CHECK_HIP(hipGetLastError());
    return THABLAS_STATUS_SUCCESS;
}

// '_h2d_ = host to device
// o, x, weight allocated on Host
// only run on 1 devices
thablasStatus_t thaDNN_h2d_s_rmsnorm_v2(float* o, float* x, float* weight, int size) 
{
    if (size==0 || o == nullptr || x == nullptr || weight == nullptr)
    {
        printf("THABLAS RMSNORM ERROR: INVALID ARGUMENT\n"); fflush(stdout);
        return THABLAS_STATUS_ALLOC_FAILED;        
    }

    int num_devices;
    CHECK_HIP(hipGetDeviceCount(&num_devices));

    if (!num_devices)
    {
        printf("THABLAS RMSNORM ERROR: COULD NOT FIND ANY COMPUTE DEVICE\n"); fflush(stdout);
        return THABLAS_STATUS_ALLOC_FAILED;
    }

    float *o_d, *x_d, *weight_d;

    CHECK_HIP(hipSetDevice(0));
    CHECK_HIP(hipMalloc(&o_d, size * sizeof(float)));
    CHECK_HIP(hipMalloc(&x_d, size * sizeof(float)));
    CHECK_HIP(hipMalloc(&weight_d, size * sizeof(float)));

    CHECK_HIP(hipMemcpy(x_d, x, size * sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(weight_d, weight, size * sizeof(float), hipMemcpyHostToDevice));

    thablasHandle_t handle;
    thablasCreate(&handle);
    thablasStatus_t status = thaDNN_s_rmsnorm_v2(handle, o_d, x_d, weight_d, size);
    if (status != THABLAS_STATUS_SUCCESS) 
    {
        printf("THABLAS RMSNORM ERROR: ERROR on Device\n"); fflush(stdout);
    }

    CHECK_HIP(hipMemcpy(o, o_d, size * sizeof(float), hipMemcpyDeviceToHost));

    CHECK_HIP(hipDeviceSynchronize());

    CHECK_HIP(hipFree(o_d));
    CHECK_HIP(hipFree(x_d));
    CHECK_HIP(hipFree(weight_d));

    return THABLAS_STATUS_SUCCESS;
}


/*
***********************************************************************************************************************************************************************************
* rmsnorm_v3 using sahed memory and reduce
***********************************************************************************************************************************************************************************
*/

// modify  to caculate sum of squares

__device__ void warpReduce_v3(volatile float *sdata, unsigned int tid) {
  int blockSize = blockDim.x;
  // printf("blockSize: %d\n", blockSize);
  if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
  if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
  if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
  if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
  if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
  if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
}

__global__ void thaDNN_s_rmsnorm_kernel_v3(float* o, float* x, float* weight, int size)
{
  int n = size;
  float *g_idata = x;
  float *g_odata = o;
  const unsigned int blockSize = blockDim.x;
  extern __shared__ float sdata[];
  
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x*(blockSize*2) + tid;
  unsigned int gridSize = blockSize*2*gridDim.x;
  sdata[tid] = 0;
  while (i < n) { 
    sdata[tid] = sdata[tid] + g_idata[i]*g_idata[i]  + g_idata[i+blockSize]*g_idata[i+blockSize]; 
    i += gridSize; }

  __syncthreads();
  if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
  if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
  if (blockSize >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }
  if (tid < 32) warpReduce_v3(sdata, tid);
  if (tid == 0) g_odata[blockIdx.x] = sdata[0];
  
  __syncthreads();
  float ss = o[0];
  ss /= size;
  ss += 1e-5f;
  ss = 1.0f / sqrtf(ss);

  for(int i = blockIdx.x * blockDim.x + threadIdx.x; 
      i < size; 
      i += blockDim.x * gridDim.x) {
    o[i] = weight[i] * (ss * x[i]);
  }

}

// '_s_' = single persion (float)
// input: o, x, weight allocated on device
// input: size = 1 -> 16384

thablasStatus_t thaDNN_s_rmsnorm_v3(thablasHandle_t handle, float* o, float* x, float* weight, int size) 
{
    if (size==0 || o == nullptr || x == nullptr || weight == nullptr || handle.current_gpu_id < 0)
    {
        printf("THABLAS RMSNORM ERROR: INVALID ARGUMENT\n"); fflush(stdout);
        return THABLAS_STATUS_ALLOC_FAILED;        
    }

    CHECK_HIP(hipSetDevice(handle.current_gpu_id));
    dim3 blockDim(512);
    dim3 gridDim(1);
    thaDNN_s_rmsnorm_kernel_v3<<<gridDim, blockDim, 512 * sizeof(float)>>>(o, x, weight, size);    
    CHECK_HIP(hipGetLastError());
    return THABLAS_STATUS_SUCCESS;
}

// '_h2d_ = host to device
// o, x, weight allocated on Host
// only run on 1 devices

thablasStatus_t thaDNN_h2d_s_rmsnorm_v3(float* o, float* x, float* weight, int size) 
{
    if (size==0 || o == nullptr || x == nullptr || weight == nullptr)
    {
        printf("THABLAS RMSNORM ERROR: INVALID ARGUMENT\n"); fflush(stdout);
        return THABLAS_STATUS_ALLOC_FAILED;        
    }

    int num_devices;
    CHECK_HIP(hipGetDeviceCount(&num_devices));

    if (!num_devices)
    {
        printf("THABLAS RMSNORM ERROR: COULD NOT FIND ANY COMPUTE DEVICE\n"); fflush(stdout);
        return THABLAS_STATUS_ALLOC_FAILED;
    }

    float *o_d, *x_d, *weight_d;

    CHECK_HIP(hipSetDevice(0));
    CHECK_HIP(hipMalloc(&o_d, size * sizeof(float)));
    CHECK_HIP(hipMalloc(&x_d, size * sizeof(float)));
    CHECK_HIP(hipMalloc(&weight_d, size * sizeof(float)));

    CHECK_HIP(hipMemcpy(x_d, x, size * sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(weight_d, weight, size * sizeof(float), hipMemcpyHostToDevice));

    thablasHandle_t handle;
    thablasCreate(&handle);
    thablasStatus_t status = thaDNN_s_rmsnorm_v3(handle, o_d, x_d, weight_d, size);
    if (status != THABLAS_STATUS_SUCCESS) 
    {
        printf("THABLAS RMSNORM ERROR: ERROR on Device\n"); fflush(stdout);
    }

    CHECK_HIP(hipMemcpy(o, o_d, size * sizeof(float), hipMemcpyDeviceToHost));

    CHECK_HIP(hipDeviceSynchronize());

    CHECK_HIP(hipFree(o_d));
    CHECK_HIP(hipFree(x_d));
    CHECK_HIP(hipFree(weight_d));

    return THABLAS_STATUS_SUCCESS;
}


/*
***************************************************************************************************************************
* matrixmulmatrix
***************************************************************************************************************************
*/

// -------------------------------------------------- matmul prefetch  -------------------------------------

template <typename T> __global__ void matmul_prefetch(T *A, T *B, T *C, int M, int K, int N) {
  /* Prefetching method.
   * Perform outer product of Asub and Bsub.
   * Specifically:
   *   Asub: TILE_SIZE * TILE_SIZE
   *   Bsub: TILE_SIZE * (TILE_SIZE * VECTOR_SIZE)
   *
   * Before calculating the submatrix, load the next TILE * TILE
   * submatrix of A into register.
   *
   * After calculating, just swap the pointer to exchange the submatrix.
   */
  int bx = blockIdx.x, by = blockIdx.y;
  int tx = threadIdx.x, ty = threadIdx.y;

  // Allocate As and next_As as column-major array
  __shared__ T As[TILE_SIZE * TILE_SIZE];
  __shared__ T next_As[TILE_SIZE * TILE_SIZE];

  // Allocate register files for sub-result of C at each thread.
  T cv[TILE_SIZE] = {0};

  // Iteration parameters is similar with
  // computational optimization method.
  int aBegin = K * TILE_SIZE * by;
  int aEnd = aBegin + K - 1;
  int aStep = TILE_SIZE;

  int bBegin = TILE_SIZE * VECTOR_SIZE * bx;
  int bStep = TILE_SIZE * N;

  int t = VECTOR_SIZE;
  T *cur = As;
  T *nxt = next_As;
  for (int i = 0; i < TILE_SIZE / VECTOR_SIZE; ++i) {
    cur[(i * t + ty) + TILE_SIZE * tx] = A[aBegin + K * (i * t + ty) + tx];
  }
  __syncthreads();

  for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) {
    // Load the next submatrix to another register files.
    // Should check the out-of-range indexing to avoid kernel crash.
    if (a + aStep <= aEnd) {
      for (int i = 0; i < TILE_SIZE / VECTOR_SIZE; ++i) {
        nxt[(i * t) + ty + TILE_SIZE * tx] = A[a + K * (i * t + ty) + tx + aStep];
      }
    }
    T *ap = cur;
    T *bp = &B[b + TILE_SIZE * ty + tx];

    for (int i = 0; i < TILE_SIZE; ++i) {
      T bv = *bp;
      for (int j = 0; j < TILE_SIZE; ++j) {
        cv[j] += ap[j] * bv;
      }
      ap += TILE_SIZE;
      bp += N;
    }
    __syncthreads();

    // Swap current submatrix and next submatrix.
    // Note that you can't directly assign nxt to cur, which
    // will change cur and nxt simultaneously at the next loop.
    T *tmp = cur;
    cur = nxt;
    nxt = tmp;
  }

  int c = N * TILE_SIZE * by + TILE_SIZE * VECTOR_SIZE * bx;
  c += TILE_SIZE * ty + tx;
  for (int i = 0; i < TILE_SIZE; ++i) {
    C[c] = cv[i];
    c += N;
  }
}

// '_s_' = single persion (float)
// input: A, B, C allocated on device
// input: M: 1 -> 32000, K: 1 -> 8192, N: 1 -> 32

thablasStatus_t thaDNN_s_matmul_prefetch(thablasHandle_t handle, float* A, float* B, float* C, int M, int K, int N) 
{
    if (M==0 || K==0 || N==0 || A == nullptr || B == nullptr || C == nullptr || handle.current_gpu_id < 0)
    {
        printf("THABLAS MATMUL ERROR: INVALID ARGUMENT\n"); fflush(stdout);
        return THABLAS_STATUS_ALLOC_FAILED;        
    }

    CHECK_HIP(hipSetDevice(handle.current_gpu_id));
    dim3 threads_prefetch(TILE_SIZE, VECTOR_SIZE);
    dim3 grid_prefetch(N / (TILE_SIZE * VECTOR_SIZE), M / TILE_SIZE);
    matmul_prefetch<float><<<grid_prefetch, threads_prefetch>>>(A, B, C, M, K, N);

    CHECK_HIP(hipGetLastError());
    return THABLAS_STATUS_SUCCESS;
}

// '_h2d_ = host to device
// A, B, C allocated on Host
// only run on 1 devices
// input: M: 1 -> 32000, K: 1 -> 8192, N: 1 -> 32

thablasStatus_t thaDNN_h2d_s_matmul_prefetch(float* A, float* B, float* C, int M, int K, int N) 
{
    if (M==0 || K==0 || N==0 || A == nullptr || B == nullptr || C == nullptr)
    {
        printf("THABLAS MATMUL ERROR: INVALID ARGUMENT\n"); fflush(stdout);
        return THABLAS_STATUS_ALLOC_FAILED;        
    }

    int num_devices;
    CHECK_HIP(hipGetDeviceCount(&num_devices));

    if (!num_devices)
    {
        printf("THABLAS MATMUL ERROR: COULD NOT FIND ANY COMPUTE DEVICE\n"); fflush(stdout);
        return THABLAS_STATUS_ALLOC_FAILED;
    }

    float *A_d, *B_d, *C_d;

    CHECK_HIP(hipSetDevice(0));
    CHECK_HIP(hipMalloc(&A_d, M * K * sizeof(float)));
    CHECK_HIP(hipMalloc(&B_d, K * N * sizeof(float)));
    CHECK_HIP(hipMalloc(&C_d, M * N * sizeof(float));

    CHECK_HIP(hipMemcpy(A_d, A, M * K * sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(B_d, B, K * N * sizeof(float), hipMemcpyHostToDevice));

    thablasHandle_t handle;
    thablasCreate(&handle);
    thablasStatus_t status = thaDNN_s_matmul_prefetch(handle, A_d, B_d, C_d, M, K, N);
    if (status != THABLAS_STATUS_SUCCESS) 
    {
        printf("THABLAS MATMUL ERROR: ERROR on Device\n"); fflush(stdout);
    }

    CHECK_HIP(hipMemcpy(C, C_d, M * N * sizeof(float), hipMemcpyDeviceToHost));

    CHECK_HIP(hipDeviceSynchronize());

    CHECK_HIP(hipFree(A_d));
    CHECK_HIP(hipFree(B_d));
    CHECK_HIP(hipFree(C_d));

    return THABLAS_STATUS_SUCCESS;
}

// -------------------------------------------------- matmul rocblas  -------------------------------------
thablasStatus_t thaDNN_s_matmul_rocblas(thablasHandle_t handle, float* A, float* B, float* C, int M, int K, int N) 
{
    if (M==0 || K==0 || N==0 || A == nullptr || B == nullptr || C == nullptr || handle.current_gpu_id < 0)
    {
        printf("THABLAS MATMUL ERROR: INVALID ARGUMENT\n"); fflush(stdout);
        return THABLAS_STATUS_ALLOC_FAILED;        
    }

    CHECK_HIP(hipSetDevice(handle.current_gpu_id));
    float alpha = 1.0;
    float beta = 0.0;
    hipblasHandle_t blas_handle;
    hipblasCreate(&blas_handle);
    rocblas_sgemm(handle, rocblas_operation_none, rocblas_operation_none, N, M, K, &alpha, d_B, N, d_A, K, &beta, d_D, N);

    CHECK_HIP(hipGetLastError());
    return THABLAS_STATUS_SUCCESS;
}


// '_h2d_ = host to device
// A, B, C allocated on Host
// only run on 1 devices
// input: M: 1 -> 32000, K: 1 -> 8192, N: 1 -> 32
thablasStatus_t thaDNN_h2d_s_matmul_rocblas(float* A, float* B, float* C, int M, int K, int N) 
{
    if (M==0 || K==0 || N==0 || A == nullptr || B == nullptr || C == nullptr)
    {
        printf("THABLAS MATMUL ERROR: INVALID ARGUMENT\n"); fflush(stdout);
        return THABLAS_STATUS_ALLOC_FAILED;        
    }

    int num_devices;
    CHECK_HIP(hipGetDeviceCount(&num_devices));

    if (!num_devices)
    {
        printf("THABLAS MATMUL ERROR: COULD NOT FIND ANY COMPUTE DEVICE\n"); fflush(stdout);
        return THABLAS_STATUS_ALLOC_FAILED;
    }

    float *A_d, *B_d, *C_d;

    CHECK_HIP(hipSetDevice(0));
    CHECK_HIP(hipMalloc(&A_d, M * K * sizeof(float));
    CHECK_HIP(hipMalloc(&B_d, K * N * sizeof(float));
    CHECK_HIP(hipMalloc(&C_d, M * N * sizeof(float));

    CHECK_HIP(hipMemcpy(A_d, A, M * K * sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(B_d, B, K * N * sizeof(float), hipMemcpyHostToDevice));

    thablasHandle_t handle;
    thablasCreate(&handle);
    thablasStatus_t status = thaDNN_s_matmul_rocblas(handle, A_d, B_d, C_d, M, K, N);
    if (status != THABLAS_STATUS_SUCCESS) 
    {
        printf("THABLAS MATMUL ERROR: ERROR on Device\n"); fflush(stdout);
    }

    CHECK_HIP(hipMemcpy(C, C_d, M * N * sizeof(float), hipMemcpyDeviceToHost));

    CHECK_HIP(hipDeviceSynchronize());

    CHECK_HIP(hipFree(A_d));
    CHECK_HIP(hipFree(B_d));
    CHECK_HIP(hipFree(C_d));

    return THABLAS_STATUS_SUCCESS;
}





/*
***************************************************************************************************************************
* forward
***************************************************************************************************************************
*/

thablasStatus_t thaDNN_s_forward(thablasHandle_t handle, Transformer* transformer, int token, int pos, float* &output_logits) {
    // a few convenience variables
    Config* p = &transformer->config;
    TransformerWeights* w = &transformer->weights;
    RunState* s = &transformer->state;
    float *x = s->x;
    int dim = p->dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads; 
    int kv_mul = p->n_heads / p->n_kv_heads; // integer multiplier of the kv sharing in multiquery
    int hidden_dim =  p->hidden_dim;
    int head_size = dim / p->n_heads;

    thablasStatus_t thablas_status = THABLAS_STATUS_SUCCESS;

    // copy the token embedding into x
    float* content_row = w->token_embedding_table + token * dim;
    memcpy(x, content_row, dim*sizeof(*x));

    // forward all the layers
    for(unsigned long long l = 0; l < p->n_layers; l++) {
        thablas_status = thaDNN_s_rmsnorm_v2(handle, s->xb, x, w->rms_att_weight + l*dim, dim);

        int loff = l * p->seq_len * kv_dim;
        s->k = s->key_cache + loff + pos * kv_dim;
        s->v = s->value_cache + loff + pos * kv_dim;

        thablas_status = thaDNN_s_matmulvec_v2(handle, s->q, s->xb, w->wq + l*dim*dim, dim, dim);
        thablas_status = thaDNN_s_matmulvec_v2(handle, s->k, s->xb, w->wk + l*dim*kv_dim, dim, kv_dim);
        thablas_status = thaDNN_s_matmulvec_v2(handle, s->v, s->xb, w->wv + l*dim*kv_dim, dim, kv_dim);

        thablas_status = thaDNN_s_rope(handle, dim, head_size, kv_dim, pos, s->q, s->k);

        // multihead attention
        thablas_status = thaDNN_s_multiheads_1(handle, pos, p->n_heads, s->q, s->att, s->key_cache, head_size, p->seq_len, loff, kv_dim, kv_mul);
        for (int h = 0; h < p->n_heads; h++) {
            float* att = s->att + h * p->seq_len;
            thablas_status = thaDNN_s_softmax(handle, att, pos + 1);
        }
        thablas_status = thaDNN_s_multiheads_3(handle, pos, p->n_heads, s->xb, s->att, s->value_cache, head_size, p->seq_len, loff, kv_dim, kv_mul, dim);
        // end multihead attention

        thablas_status = thaDNN_s_matmulvec_v2(handle, s->xb2, s->xb, w->wo + l*dim*dim, dim, dim);

        thablas_status = thaBLAS_s_vecaddvec(handle, x, s->xb2, dim);

        thablas_status = thaDNN_s_rmsnorm_v2(handle, s->xb, x, w->rms_ffn_weight + l*dim, dim);

        thablas_status = thaDNN_s_matmulvec_v2(handle, s->hb, s->xb, w->w1 + l*dim*hidden_dim, dim, hidden_dim);
        thablas_status = thaDNN_s_matmulvec_v2(handle, s->hb2, s->xb, w->w3 + l*dim*hidden_dim, dim, hidden_dim);

        thablas_status = thaDNN_s_swiglu(handle, s->hb, s->hb2, hidden_dim);

        thablas_status = thaDNN_s_matmulvec_v2(handle, s->xb, s->hb, w->w2 + l*dim*hidden_dim, hidden_dim, dim);

        thablas_status = thaBLAS_s_vecaddvec(handle, x, s->xb, dim);
    }

    thablas_status = thaDNN_s_rmsnorm_v2(handle, x, x, w->rms_final_weight, dim);

    thablas_status = thaDNN_s_matmulvec_v2(handle, s->logits, x, w->wcls, p->dim, p->vocab_size);
    
    output_logits = s->logits;
    return thablas_status;
}
