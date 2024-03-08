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
__global__ void thaDNN_s_softmax_kernel(float* output, float* x, int size)
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
            output[j_pos + i] = local_x[i] / sum;
    }

}

// _s_ = single persion (float)
// input: output, x allocated on device
// input: size = 32000
thablasStatus_t thaDNN_s_softmax(thablasHandle_t handle, float* output, float* x, int size) 
{
    if (size==0 || output == nullptr || x == nullptr || handle.current_gpu_id < 0)
    {
        printf("THABLAS SOFTMAX ERROR: INVALID ARGUMENT\n"); fflush(stdout);
        return THABLAS_STATUS_ALLOC_FAILED;        
    }

    CHECK_HIP(hipSetDevice(handle.current_gpu_id));
    dim3 blockDim((size + 32 - 1) / 32);
    dim3 gridDim(1);
    // dim3 gridSize((size + SOFTMAX_BLOCK_SIZE - 1) / SOFTMAX_BLOCK_SIZE);
    hipLaunchKernelGGL(thaDNN_s_softmax_kernel, gridDim, blockDim, 0, 0, output, x, size);
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

    float *output_d, *x_d;

    CHECK_HIP(hipSetDevice(0));
    CHECK_HIP(hipMalloc(&output_d, size * sizeof(float)));
    CHECK_HIP(hipMalloc(&x_d, size * sizeof(float)));

    CHECK_HIP(hipMemcpy(x_d, x, size * sizeof(float), hipMemcpyHostToDevice));

    thablasHandle_t handle;
    thablasCreate(&handle);
    thablasStatus_t status = thaDNN_s_softmax(handle, output_d, x_d, size);
    if (status != THABLAS_STATUS_SUCCESS) {
        printf("THABLAS SOFTMAX ERROR: ERROR on Device\n"); fflush(stdout);
    }

    CHECK_HIP(hipMemcpy(x, output_d, size * sizeof(float), hipMemcpyDeviceToHost));

    CHECK_HIP(hipDeviceSynchronize());

    CHECK_HIP(hipFree(output_d));
    CHECK_HIP(hipFree(x_d));

    return THABLAS_STATUS_SUCCESS;
}


/*
*********************************************************************************************************
* forward
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
  // CHECK_HIP(hipMemset(s_xb, 0, dim * sizeof(float)));
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
      memset(s->xb, 0, dim * sizeof(float));
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
    for (int i = 0; i < hidden_dim; i++) {
      float val = s->hb[i];
      // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
      val *= (1.0f / (1.0f + expf(-val)));
      // elementwise multiply with w3(x)
      val *= s->hb2[i];
      s->hb[i] = val;
    }

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
navie code:
void rmsnorm(float* o, float* x, float* weight, int size) {
  // calculate sum of squares
  float ss = 0.0f;
  for (int j = 0; j < size; j++) {
    ss += x[j] * x[j];
  }
  ss /= size;
  ss += 1e-5f;
  ss = 1.0f / sqrtf(ss);
  // normalize and scale
  for (int j = 0; j < size; j++) {
    o[j] = weight[j] * (ss * x[j]);
  }
}
*/


// biến warpSize là biến built-in của HIP, mặc định là 64
__inline__ __device__
int warpReduceSum(int val) {
  // printf("warpSize: %d\n", warpSize);
  for (int offset = warpSize/2; offset > 0; offset /= 2) 
    val += __shfl_down(val, offset);
  return val;
}


__inline__ __device__
int blockReduceSum(int val) {

  static __shared__ int shared[64]; // Shared mem for 64 partial sums
  int lane = threadIdx.x % warpSize;
  int wid = threadIdx.x / warpSize;

  val = warpReduceSum(val);     // Each warp performs partial reduction

  if (lane==0) shared[wid]=val; // Write reduced value to shared memory

  __syncthreads();              // Wait for all partial reductions

  //read from shared memory only if that warp existed
  val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;

  if (wid==0) val = warpReduceSum(val); //Final reduce within first warp

  return val;
}

__global__ void deviceReduceWarpAtomicKernel(int *in, int* out, int N) {
  int sum = int(0);
  for(int i = blockIdx.x * blockDim.x + threadIdx.x; 
      i < N; 
      i += blockDim.x * gridDim.x) {
    sum += in[i];
  }
  sum = warpReduceSum(sum);
  if ((threadIdx.x & (warpSize - 1)) == 0)
    atomicAdd(out, sum);
}


// modify deviceReduceBlockAtomicKernel to caculate sum of squares
__global__ void thaDNN_s_rmsnorm_kernel_v2(float* o, float* x, float* weight, int size)
{
    const int thread_id = threadIdx.x;
    const int block_id = blockIdx.x;
    const int block_size = blockDim.x;

    float ss=0.0f;
    for (int elem_id = thread_id; elem_id < size; elem_id += block_size){
        ss += x[elem_id] * x[elem_id];
    }

    // reduce ss
    ss 
}

