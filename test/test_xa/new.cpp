#include <cstdio>
#include <mma.h>

#include "matmul.h"

#define WARPSIZE 32
#define FRAGSIZE 16
#define BLOCKSIZE 4

using namespace nvcuda;

#define CHECK_CUDA(call)                                                 \
  do {                                                                   \
    cudaError_t status_ = call;                                          \
    if (status_ != cudaSuccess) {                                        \
      fprintf(stderr, "CUDA error (%s:%d): %s:%s\n", __FILE__, __LINE__, \
              cudaGetErrorName(status_), cudaGetErrorString(status_));   \
      exit(EXIT_FAILURE);                                                \
    }                                                                    \
  } while (0)

// Device(GPU) pointers
static half *A_gpu, *B_gpu;
static float *C_gpu;

void naive_cpu_matmul(half *_A, half *_B, float *_C, int M, int N, int K) {
  for (int i = 0; i < M; i++) {
    for (int k = 0; k < K; k++) {
      for (int j = 0; j < N; j++) {
        _C[i * N + j] += (float)_A[i * K + k] * (float)_B[k * N + j];
      }
    }
  }
}

__global__ void gpu_matmul(half *A, half *B, float *C, int M, int N, int K) {
  
  using namespace wmma;

  int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int global_idy = blockIdx.y * blockDim.y + threadIdx.y;

  int j = global_idx / WARPSIZE * FRAGSIZE;
  int i = global_idy * FRAGSIZE;

  int local_j = threadIdx.x / WARPSIZE;
  int local_i = threadIdx.y;
  int line = threadIdx.x % WARPSIZE;

  fragment<matrix_a, FRAGSIZE, FRAGSIZE, FRAGSIZE, half, row_major> a_frag;
  fragment<matrix_b, FRAGSIZE, FRAGSIZE, FRAGSIZE, half, row_major> b_frag;
  fragment<accumulator, FRAGSIZE, FRAGSIZE, FRAGSIZE, float> c_frag;

  fill_fragment(c_frag, 0.0f);

  __shared__ half a[BLOCKSIZE][FRAGSIZE][FRAGSIZE];
  __shared__ half b[BLOCKSIZE][FRAGSIZE][FRAGSIZE];

  int ii = local_j * 4 + line / 8;
  int jj = local_i * 4 + line / 8;
  int kk = line % 8 * 2 + 0;
  for (int k = 0; k < K; k += FRAGSIZE) {
    a[local_i][ii][kk + 0] = A[(i + ii) * K + (k + kk + 0)];
    a[local_i][ii][kk + 1] = A[(i + ii) * K + (k + kk + 1)];
    b[local_j][jj][kk + 0] = B[(k + jj) * N + (j + kk + 0)];
    b[local_j][jj][kk + 1] = B[(k + jj) * N + (j + kk + 1)];
    __syncthreads();
    load_matrix_sync(a_frag, &a[local_i][0][0], FRAGSIZE);
    load_matrix_sync(b_frag, &b[local_j][0][0], FRAGSIZE);
    mma_sync(c_frag, a_frag, b_frag, c_frag);
    __syncthreads();
  }

  store_matrix_sync(&C[i * N + j], c_frag, N, mem_row_major);
}

void matmul(half *_A, half *_B, float *_C, int M, int N, int K) {

  // (TODO) Upload A and B matrix to GPU
  CHECK_CUDA(cudaMemcpy(A_gpu, _A, sizeof(half) * M * K, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(B_gpu, _B, sizeof(half) * K * N, cudaMemcpyHostToDevice));

  // (TODO) Launch kernel on a GPU
  dim3 blockDim(BLOCKSIZE * WARPSIZE, BLOCKSIZE);
  dim3 gridDim((N + (FRAGSIZE * blockDim.x / WARPSIZE) - 1) / (FRAGSIZE * blockDim.x / WARPSIZE), (M + (FRAGSIZE * blockDim.y) - 1) / (FRAGSIZE * blockDim.y));
  gpu_matmul<<<gridDim, blockDim>>>(A_gpu, B_gpu, C_gpu, M, N, K);
  CHECK_CUDA(cudaGetLastError());

  // (TODO) Download C matrix from GPU
  CHECK_CUDA(cudaMemcpy(_C, C_gpu, sizeof(float) * M * N, cudaMemcpyDeviceToHost));

  // DO NOT REMOVE; NEEDED FOR TIME MEASURE
  CHECK_CUDA(cudaDeviceSynchronize());
}

void matmul_init(int M, int N, int K) {
  // (TODO) Allocate device memory
  CHECK_CUDA(cudaMalloc(&A_gpu, sizeof(half) * M * K));
  CHECK_CUDA(cudaMalloc(&B_gpu, sizeof(half) * K * N));
  CHECK_CUDA(cudaMalloc(&C_gpu, sizeof(float) * M * N));

  // DO NOT REMOVE; NEEDED FOR TIME MEASURE
  CHECK_CUDA(cudaDeviceSynchronize());
}

void matmul_cleanup(half *_A, half *_B, float *_C, int M, int N, int K) {
  // (TODO) Do any post-matmul cleanup work here.
  CHECK_CUDA(cudaFree(A_gpu));
  CHECK_CUDA(cudaFree(B_gpu));
  CHECK_CUDA(cudaFree(C_gpu));

  // DO NOT REMOVE; NEEDED FOR TIME MEASURE
  CHECK_CUDA(cudaDeviceSynchronize());
}
