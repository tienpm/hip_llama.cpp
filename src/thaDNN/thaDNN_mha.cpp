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

__global__ void thaDNN_s_multiheads_1_v2_batch_kernel(int pos[], int n_heads, int n_layers, float* s_q_batch, float* s_att_batch, float* s_key_cache_batch, int head_size, int p_seq_len, int loff, int dim, int kv_dim, int kv_mul) {
    int lx = threadIdx.x;
    int gx = blockIdx.x;

    int b = 0;
    int pos_b = pos[b];
    while (gx >= ((pos_b+1) * n_heads))
    {
        gx -= (pos_b+1) * n_heads;
        ++b;
        pos_b = pos[b];
    }

    int t = gx % (pos_b + 1);
    int h = gx / (pos_b + 1);

    float* s_q = s_q_batch + b * dim;
    float* s_att = s_att_batch + b * n_heads * p_seq_len;
    float* s_key_cache = s_key_cache_batch + b * n_layers * p_seq_len * kv_dim;

    float score = 0.0f;
    float* q = s_q + h * head_size;
    float* k = s_key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
    float* att = s_att + h * p_seq_len;
    for(int i=lx ; i<head_size ; i+=blockDim.x)
    {
        score += q[i] * k[i];
    }

    score = block_reduce_sum(score);
    if (lx==0)
    {
        att[t] = score / sqrtf(head_size);
    }
}

thablasStatus_t thaDNN_s_multiheads_1_v2_batch(thablasHandle_t handle, int n_batches, int pos[], int pos_d[], int n_heads, int n_layers, float* s_q_batch, float* s_att_batch, float* s_key_cache_batch, int head_size, int seq_len, int loff, int kv_dim, int dim, int kv_mul) {
    // if (s_q_batch==nullptr || s_att_batch==nullptr || s_key_cache_batch==nullptr || head_size + seq_len + kv_dim + dim==0)
    // {
    //     printf("THABLAS MULTI_HEADS_1 BATCH ERROR: INVALID ARGUMENT\n"); fflush(stdout);
    //     return THABLAS_STATUS_ALLOC_FAILED;        
    // }

    int total_poses = 0;
    for(int b=0 ; b<n_batches ; ++b)
    {
        total_poses += (pos[b]+1);
    }

    // CHECK_HIP(hipSetDevice(handle.current_gpu_id));


    dim3 blockDim(MAX_BLOCK_SIZE);
    dim3 gridDim(total_poses * n_heads);
    // CAUTION: careful playing with [pos]. 
    hipLaunchKernelGGL(thaDNN_s_multiheads_1_v2_batch_kernel, gridDim, blockDim, 0, 0, pos_d, n_heads, n_layers, s_q_batch, s_att_batch, s_key_cache_batch, head_size, seq_len, loff, kv_dim, dim, kv_mul);
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
    for (int i=lx ; i<size ; i+=bDim)
    {
        private_max_val = std::max(private_max_val, x[i]);
    }

    private_max_val = block_reduce_max(private_max_val);
    if (lx==0)
    {
        max_val = private_max_val;
    }
    __syncthreads();
    private_max_val = max_val;
    
    float private_sum = 0.0f, tmp;
    __shared__ float sum;
    for (int i =lx; i<size ; i+=bDim) {
        tmp = expf(x[i] - private_max_val);
        x[i] = tmp;
        private_sum += tmp;
    }

    private_sum = block_reduce_sum(private_sum);
    if (lx==0)
    {
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



__global__ void thaDNN_s_multiheads_3_v2_batch_kernel(int pos[], int n_heads, float *s_xb_batch, float *s_att_batch, float *s_value_cache_batch, int head_size, int seq_len, int loff, int kv_dim, int kv_mul, int dim, int n_layers) {
    int lx = threadIdx.x;

    int i = blockIdx.x;
    int h = blockIdx.y;
    int b = blockIdx.z;

    float sum = 0.0f;
    float *att, *v, *xb;
    int pos_b = pos[b];
    for(int t=lx ; t<pos_b+1 ; t+=blockDim.x)
    {
        att = s_att_batch + h * seq_len + b * n_heads *  seq_len;
        float a = att[t];

        v = s_value_cache_batch + loff + t * kv_dim + (h / kv_mul) * head_size + b * n_layers * seq_len * kv_dim;

        sum += a * v[i];
    }
    sum = block_reduce_sum(sum);
    if (lx == 0)
    {
        xb = s_xb_batch + h * head_size + b * dim;
        xb[i] = sum;
    }
}

thablasStatus_t thaDNN_s_multiheads_3_v2_batch(thablasHandle_t handle, int n_batches, int pos_d[], int n_heads, float *s_xb_batch, float *s_att_batch, float *s_value_cache_batch, int head_size, int seq_len, int loff, int kv_dim, int kv_mul, int dim, int n_layers) {
    // if (s_xb_batch==nullptr || s_att_batch==nullptr || s_value_cache_batch==nullptr || head_size==0 || seq_len==0 || kv_dim==0)
    // {
    //     printf("THABLAS MULTI_HEADS_3 BATCH ERROR: INVALID ARGUMENT\n"); fflush(stdout);
    //     return THABLAS_STATUS_ALLOC_FAILED;        
    // }

    // CHECK_HIP(hipSetDevice(handle.current_gpu_id));
    // CHECK_HIP(hipMemset(s_xb_batch, 0, n_batches * dim * sizeof(float)));
    dim3 blockDim(1024);
    dim3 gridDim(head_size, n_heads, n_batches);
    // CAUTION: careful playing with [pos]. 
    hipLaunchKernelGGL(thaDNN_s_multiheads_3_v2_batch_kernel, gridDim, blockDim, 0, 0, pos_d, n_heads, s_xb_batch, s_att_batch, s_value_cache_batch, head_size, seq_len, loff, kv_dim, kv_mul, dim, n_layers);
    // CHECK_HIP(hipGetLastError());

    return THABLAS_STATUS_SUCCESS;
}

__global__ void thaDNN_s_matmulvec_v2_batch_kernel(float *C_batch, float *B_batch, float *A, int K, int M, int Coff, int has_pos, int pos[], int C_batch_size, int B_batch_size) {
    int gx = blockIdx.x;
    int b = blockIdx.y;
    int lx = threadIdx.x;
    float sum = 0.0f;

    float *C = C_batch + Coff + has_pos * pos[b] + b * C_batch_size;
    float *B = B_batch + b * B_batch_size;
    for (int k=lx ; k<K ; k+=blockDim.x)
    {
        sum += A[gx*K + k] * B[k];
    }
    sum = block_reduce_sum(sum);
    if (lx == 0)
    {
        C[gx] = sum;
    }
}

// A[M,K] x B[K,1] = C[1,M]
thablasStatus_t thaDNN_s_matmulvec_v2_batch(thablasHandle_t handle, int n_batches, float *C_batch, float *B_batch, float *A, int K, int M, int Coff, int has_pos, int pos_d[], int C_batch_size, int B_batch_size) {
    // if (K + M + n_batches==0 || A == nullptr || B_batch == nullptr || C_batch == nullptr || handle.current_gpu_id < 0)
    // {
    //     printf("THABLAS MAT MUL VEC BATCH ERROR: INVALID ARGUMENT\n"); fflush(stdout);
    //     return THABLAS_STATUS_ALLOC_FAILED;        
    // }

    // CHECK_HIP(hipSetDevice(handle.current_gpu_id));
    dim3 blockDim(MAX_BLOCK_SIZE);
    dim3 gridDim(M, n_batches);

    hipLaunchKernelGGL(thaDNN_s_matmulvec_v2_batch_kernel, gridDim, blockDim, 0, 0, C_batch, B_batch, A, K, M, Coff, has_pos, pos_d, C_batch_size, B_batch_size);
    // CHECK_HIP(hipGetLastError());

    return THABLAS_STATUS_SUCCESS;
}

