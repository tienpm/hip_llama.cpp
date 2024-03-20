#include "thaDNN/thaDNN_mha.hpp"

#define WARP_SIZE 64
#define MAX_BLOCK_SIZE 1024

__device__ float warp_reduce_sum(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) 
        val += __shfl_xor(val, offset);
    return val;
}

__device__ float block_reduce_sum(float val) {
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

__device__ float warp_reduce_max(float val) {
  for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
    val = std::max(val, __shfl_xor(val, offset));
  return val;
}

__device__ float block_reduce_max(float val) {
  static __shared__ float shared[MAX_BLOCK_SIZE / WARP_SIZE];
  int lane = threadIdx.x % WARP_SIZE;
  int wid = threadIdx.x / WARP_SIZE;

  val = warp_reduce_max(val);

  if (lane == 0) 
    shared[wid] = val;

  __syncthreads();

  val = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared[lane] : -3.402e+38;

  if (wid == 0) val = warp_reduce_max(val);

  return val;
}

__global__ void thaDNN_s_multiheads_1_v2_batch_kernel(int pos[], int n_heads, int pipe_size, int batch_size, float* s_q_batch, float* s_att_batch, float* s_key_cache_batch, int head_size, int n_words, int kv_dim, int dim, int kv_mul) {
    int lx = threadIdx.x;
    int gx = blockIdx.x;

    int b = 0;
    int pos_b = pos[b];
    while (gx >= ((pos_b+1) * n_heads)) {
        gx -= (pos_b+1) * n_heads;
        ++b;
        pos_b = pos[b];
    }

    int t = gx % (pos_b + 1);
    int h = gx / (pos_b + 1);

    float* s_q = s_q_batch + b * dim;
    float* s_att = s_att_batch + b * n_heads * n_words;
    // float* s_key_cache = s_key_cache_batch + b * n_layers * p_seq_len * kv_dim;
    // float* s_k = s_key_cache_batch + t * (pipe_size * batch_size * kv_dim) + l * batch_size * kv_dim + b * kv_dim;
    float* s_k = s_key_cache_batch + t * batch_size * kv_dim + b * kv_dim;

    // k = s_key_cache + loff + t * KV_DIM + (h / KV_MUL) * HEAD_SIZE
    float score = 0.0f;
    float* q = s_q + h * head_size;
    // float* k = s_key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
    float* k = s_k + (h / kv_mul) * head_size;
    float* att = s_att + h * n_words;
    for(int i = lx ; i < head_size ; i += blockDim.x) {
        score += q[i] * k[i];
    }

    score = block_reduce_sum(score);
    if (lx==0) {
        att[t] = score / sqrtf(head_size);
    }
}

thablasStatus_t thaDNN_s_multiheads_1_v2_batch(thablasHandle_t handle, int batch_size, int pipe_size, int pos[], int pos_d[], int n_heads, float* s_q_batch, float* s_att_batch, float* s_key_cache_batch, int head_size, int n_words, int kv_dim, int dim, int kv_mul) {
    // if (s_q_batch==nullptr || s_att_batch==nullptr || s_key_cache_batch==nullptr || head_size + seq_len + kv_dim + dim==0)
    // {
    //     printf("THABLAS MULTI_HEADS_1 BATCH ERROR: INVALID ARGUMENT\n"); fflush(stdout);
    //     return THABLAS_STATUS_ALLOC_FAILED;        
    // }

    int total_poses = 0;
    for(int b = 0 ; b < batch_size ; ++b) {
        total_poses += (pos[b]+1);
    }

    // CHECK_HIP(hipSetDevice(handle.current_gpu_id));


    dim3 blockDim(MAX_BLOCK_SIZE);
    dim3 gridDim(total_poses * n_heads);
    // CAUTION: careful playing with [pos]. 
    hipLaunchKernelGGL(thaDNN_s_multiheads_1_v2_batch_kernel, gridDim, blockDim, 0, 0, pos_d, n_heads, pipe_size, batch_size, s_q_batch, s_att_batch, s_key_cache_batch, head_size, n_words, kv_dim, dim, kv_mul);
    // CHECK_HIP(hipGetLastError());

    return THABLAS_STATUS_SUCCESS;
}



__global__ void thaDNN_s_multiheads_2_batch_kernel(int n_batches, float* s_att_batch, int size_batch[], int seq_len, int n_heads) {
    int lx = threadIdx.x;
    int bDim = blockDim.x;
    int h = blockIdx.x;
    int b = blockIdx.y;

    float* s_att = s_att_batch + b * n_heads * seq_len;
    float* x = s_att + h * seq_len;
    int size = size_batch[b] + 1;

    float private_max_val = -3.402e+38;
    __shared__ float max_val;
    for (int i=lx ; i<size ; i+=bDim) {
        private_max_val = std::max(private_max_val, x[i]);
    }

    private_max_val = block_reduce_max(private_max_val);
    if (lx == 0) {
        max_val = private_max_val;
    }
    __syncthreads();
    private_max_val = max_val;
    
    float private_sum = 0.0f, tmp;
    __shared__ float sum;
    for (int i = lx; i < size ; i += bDim) {
        tmp = expf(x[i] - private_max_val);
        x[i] = tmp;
        private_sum += tmp;
    }

    private_sum = block_reduce_sum(private_sum);
    if (lx == 0) {
        sum = private_sum;
    }
    __syncthreads();
    private_sum = sum;

    for (int i =lx; i<size ; i+=bDim) {
        x[i] /= private_sum;
    }
}

// _s_ = single persion (float)
// input: output, x allocated on device
// input: size = 32000
thablasStatus_t thaDNN_s_multiheads_2_batch(thablasHandle_t handle, int n_batches, float* s_att_batch, int size_batch[], int seq_len, int n_heads) {
    // if (seq_len+n_heads+n_batches==0 || s_att_batch == nullptr || handle.current_gpu_id < 0)
    // {
    //     printf("THABLAS SOFTMAX BATCH ERROR: INVALID ARGUMENT\n"); fflush(stdout);
    //     return THABLAS_STATUS_ALLOC_FAILED;        
    // }

    // CHECK_HIP(hipSetDevice(handle.current_gpu_id));
    
    dim3 blockDim(1024);
    dim3 gridDim(n_heads, n_batches);

    hipLaunchKernelGGL(thaDNN_s_multiheads_2_batch_kernel, gridDim, blockDim, 0, 0, n_batches, s_att_batch, size_batch, seq_len, n_heads);
    // CHECK_HIP(hipGetLastError());
    return THABLAS_STATUS_SUCCESS;
}



__global__ void thaDNN_s_multiheads_3_v2_batch_kernel(int pos[], int n_heads, int batch_size, float *s_xb_batch, float *s_att_batch, float *s_value_cache_batch, int head_size, int n_words, int kv_dim, int kv_mul, int dim, int pipe_size) {
    int lx = threadIdx.x;

    int i = blockIdx.x;
    int h = blockIdx.y;
    int b = blockIdx.z;

    float sum = 0.0f;
    float *att, *v, *xb;
    int pos_b = pos[b];
    for(int t = lx ; t < pos_b + 1 ; t += blockDim.x) {
        att = s_att_batch + h * n_words + b * n_heads *  n_words;
        float a = att[t];

        // v = s_value_cache_batch + loff + t * kv_dim + (h / kv_mul) * head_size + b * n_layers * seq_len * kv_dim;
        // v = s_value_cache_batch + t * (pipe_size * batch_size * kv_dim) + l * batch_size * kv_dim + b * kv_dim + (h / kv_mul) * head_size;
        v = s_value_cache_batch + t * batch_size * kv_dim + b * kv_dim + (h / kv_mul) * head_size;

        sum += a * v[i];
    }

    sum = block_reduce_sum(sum);
    if (lx == 0) {
        xb = s_xb_batch + h * head_size + b * dim;
        xb[i] = sum;
    }
}

thablasStatus_t thaDNN_s_multiheads_3_v2_batch(thablasHandle_t handle, int batch_size, int pos_d[], int n_heads, float *s_xb_batch, float *s_att_batch, float *s_value_cache_batch, int head_size, int n_words, int kv_dim, int kv_mul, int dim, int pipe_size) {
    // if (s_xb_batch==nullptr || s_att_batch==nullptr || s_value_cache_batch==nullptr || head_size==0 || seq_len==0 || kv_dim==0)
    // {
    //     printf("THABLAS MULTI_HEADS_3 BATCH ERROR: INVALID ARGUMENT\n"); fflush(stdout);
    //     return THABLAS_STATUS_ALLOC_FAILED;        
    // }

    // CHECK_HIP(hipSetDevice(handle.current_gpu_id));
    // CHECK_HIP(hipMemset(s_xb_batch, 0, n_batches * dim * sizeof(float)));
    dim3 blockDim(1024);
    dim3 gridDim(head_size, n_heads, batch_size);
    // CAUTION: careful playing with [pos]. 
    hipLaunchKernelGGL(thaDNN_s_multiheads_3_v2_batch_kernel, gridDim, blockDim, 0, 0, pos_d, n_heads, batch_size, s_xb_batch, s_att_batch, s_value_cache_batch, head_size, n_words, kv_dim, kv_mul, dim, pipe_size);
    // CHECK_HIP(hipGetLastError());

    return THABLAS_STATUS_SUCCESS;
}
