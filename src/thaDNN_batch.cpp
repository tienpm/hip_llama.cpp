#include "thaDNN.hpp"
#include "thaBLAS.hpp"
#include "hip_helper.hpp"
#include "seq.hpp"

#include <hip/hip_runtime.h>
#include <omp.h>

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

__global__ void thaDNN_s_multiheads_1_v2_batch_kernel(int pos[], int n_heads, int pipe_size, int batch_size, float* s_q_batch, float* s_att_batch, float* s_key_cache_batch, int head_size, int n_words, int kv_dim, int dim, int kv_mul)
{
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

thablasStatus_t thaDNN_s_multiheads_1_v2_batch(thablasHandle_t handle, int batch_size, int pipe_size, int pos[], int pos_d[], int n_heads, float* s_q_batch, float* s_att_batch, float* s_key_cache_batch, int head_size, int n_words, int kv_dim, int dim, int kv_mul)
{
    // if (s_q_batch==nullptr || s_att_batch==nullptr || s_key_cache_batch==nullptr || head_size + seq_len + kv_dim + dim==0)
    // {
    //     printf("THABLAS MULTI_HEADS_1 BATCH ERROR: INVALID ARGUMENT\n"); fflush(stdout);
    //     return THABLAS_STATUS_ALLOC_FAILED;        
    // }

    int total_poses = 0;
    for(int b=0 ; b<batch_size ; ++b)
    {
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

__global__ void thaDNN_s_rmsnorm_kernel_v2_batch(int n_batches, float* o_batch, float* x_batch, float* weight, int size, int dim)
{
    int lx = threadIdx.x;
    int b = blockIdx.x;
    float tmp;
    float ss = 0.0;
    
    float* x = x_batch + b * dim;
    float* o = o_batch + b * dim;
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

thablasStatus_t thaDNN_s_rmsnorm_v2_batch(thablasHandle_t handle, int n_batches, float* o_batch, float* x_batch, float* weight, int size, int dim) 
{
    // if (size+dim==0 || o_batch == nullptr || x_batch == nullptr || weight == nullptr || handle.current_gpu_id < 0)
    // {
    //     printf("THABLAS RMSNORM V2 BATCH ERROR: INVALID ARGUMENT\n"); fflush(stdout);
    //     return THABLAS_STATUS_ALLOC_FAILED;        
    // }

    // CHECK_HIP(hipSetDevice(handle.current_gpu_id));
    dim3 blockDim(1024);
    dim3 gridDim(n_batches);
    hipLaunchKernelGGL(thaDNN_s_rmsnorm_kernel_v2_batch, gridDim, blockDim, 0, 0, n_batches, o_batch, x_batch, weight, size, dim);
    
    // CHECK_HIP(hipGetLastError());
    return THABLAS_STATUS_SUCCESS;
}


__global__ void thaDNN_s_multiheads_2_batch_kernel(int n_batches, float* s_att_batch, int size_batch[], int seq_len, int n_heads) 
{
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
thablasStatus_t thaDNN_s_multiheads_2_batch(thablasHandle_t handle, int n_batches, float* s_att_batch, int size_batch[], int seq_len, int n_heads)
{
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

__global__ void thaDNN_s_matmulvec_v2_batch_kernel(float *C_batch, float *B_batch, float *A, int K, int M, int Coff, int has_pos, int pos[], int C_batch_size, int B_batch_size)
{
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
thablasStatus_t thaDNN_s_matmulvec_v2_batch(thablasHandle_t handle, int n_batches, float *C_batch, float *B_batch, float *A, int K, int M, int Coff, int has_pos, int pos_d[], int C_batch_size, int B_batch_size)
{
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


__global__ void thaDNN_s_multiheads_3_v2_batch_kernel(int pos[], int n_heads, int batch_size, float *s_xb_batch, float *s_att_batch, float *s_value_cache_batch, int head_size, int n_words, int kv_dim, int kv_mul, int dim, int pipe_size)
{
    int lx = threadIdx.x;

    int i = blockIdx.x;
    int h = blockIdx.y;
    int b = blockIdx.z;

    float sum = 0.0f;
    float *att, *v, *xb;
    int pos_b = pos[b];
    for(int t=lx ; t<pos_b+1 ; t+=blockDim.x)
    {
        att = s_att_batch + h * n_words + b * n_heads *  n_words;
        float a = att[t];

        // v = s_value_cache_batch + loff + t * kv_dim + (h / kv_mul) * head_size + b * n_layers * seq_len * kv_dim;
        // v = s_value_cache_batch + t * (pipe_size * batch_size * kv_dim) + l * batch_size * kv_dim + b * kv_dim + (h / kv_mul) * head_size;
        v = s_value_cache_batch + t * batch_size * kv_dim + b * kv_dim + (h / kv_mul) * head_size;

        sum += a * v[i];
    }
    sum = block_reduce_sum(sum);
    if (lx == 0)
    {
        xb = s_xb_batch + h * head_size + b * dim;
        xb[i] = sum;
    }
}

thablasStatus_t thaDNN_s_multiheads_3_v2_batch(thablasHandle_t handle, int batch_size, int pos_d[], int n_heads, float *s_xb_batch, float *s_att_batch, float *s_value_cache_batch, int head_size, int n_words, int kv_dim, int kv_mul, int dim, int pipe_size)
{
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


// __global__ void thaDNN_s_matmulvec_v2_batch_kv_cache_kernel(float *C_batch, float *B_batch, float *A, int K, int M, int pos[], int B_batch_size, int pipe_size, int batch_size, int kv_dim, int l)
// {
//     int gx = blockIdx.x;
//     int b = blockIdx.y;
//     int lx = threadIdx.x;
//     float sum = 0.0f;

//     // C = key_cache/value_cache
//     // A = W;
//     // B = Sx
//     float *C = C_batch + pos[b] * (pipe_size * batch_size * kv_dim) + l * batch_size * kv_dim + b * kv_dim;
//     float *B = B_batch + b * B_batch_size;
//     for (int k=lx ; k<K ; k+=blockDim.x)
//     {
//         sum += A[gx*K + k] * B[k];
//     }
//     sum = block_reduce_sum(sum);
//     if (lx == 0)
//     {
//         C[gx] = sum;
//     }
// }

// A[M,K] x B[K,1] = C[1,M]
// thablasStatus_t thaDNN_s_matmulvec_v2_batch_kv_cache(thablasHandle_t handle, int batch_size, float *C_batch, float *B_batch, float *A, int K, int M, int pos[], int B_batch_size, int pipe_size, int kv_dim, int l)
// {
//     // if (K + M + n_batches==0 || A == nullptr || B_batch == nullptr || C_batch == nullptr || handle.current_gpu_id < 0)
//     // {
//     //     printf("THABLAS MAT MUL VEC BATCH ERROR: INVALID ARGUMENT\n"); fflush(stdout);
//     //     return THABLAS_STATUS_ALLOC_FAILED;        
//     // }

//     // CHECK_HIP(hipSetDevice(handle.current_gpu_id));
//     dim3 blockDim(MAX_BLOCK_SIZE);
//     dim3 gridDim(M, batch_size);

//     hipLaunchKernelGGL(thaDNN_s_matmulvec_v2_batch_kv_cache_kernel, gridDim, blockDim, 0, 0, C_batch, B_batch, A, K, M, pos, B_batch_size, pipe_size, batch_size, kv_dim, l);
//     // CHECK_HIP(hipGetLastError());

//     return THABLAS_STATUS_SUCCESS;
// }

// B and C are col major
__global__ void thaDNN_s_matmul_batch_kernel(float *A, float *B, float *C, int M, int N, int K)
{
    int i = blockIdx.x;
    int j = blockIdx.y;
    int lx = threadIdx.x;
    float sum = 0.0f;

    float *Ccol = C + j * M;
    float *Bcol = B + j * K;
    for (int k=lx ; k<K ; k+=blockDim.x)
    {
        sum += A[i*K + k] * Bcol[k];
    }
    sum = block_reduce_sum(sum);
    if (lx == 0)
    {
        Ccol[i] = sum;
    }
}


// B and C are col major
thablasStatus_t thaDNN_s_matmul_batch(thablasHandle_t handle, float *A, float *B, float *C, int M, int N, int K)
{
    // if (K + M + n_batches==0 || A == nullptr || B_batch == nullptr || C_batch == nullptr || handle.current_gpu_id < 0)
    // {
    //     printf("THABLAS MAT MUL VEC BATCH ERROR: INVALID ARGUMENT\n"); fflush(stdout);
    //     return THABLAS_STATUS_ALLOC_FAILED;        
    // }

    // CHECK_HIP(hipSetDevice(handle.current_gpu_id));
    dim3 blockDim(MAX_BLOCK_SIZE);
    dim3 gridDim(M, N);

    hipLaunchKernelGGL(thaDNN_s_matmul_batch_kernel, gridDim, blockDim, 0, 0, A, B, C, M, N ,K);
    // CHECK_HIP(hipGetLastError());

    return THABLAS_STATUS_SUCCESS;
}