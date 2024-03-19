#include <hip/hip_runtime.h>

#include <chrono>
// #include <rocblas.h>
#include <rocblas/rocblas.h>
#include <stdio.h>

#include <iostream>


int M = 32000;
int K = 8192;
int N = 128;  // N >= TILE_SIZE * TILE_SIZE
const int TILE_SIZE = 32; // TILE_SIZE >= VECTOR_SIZE
const int VECTOR_SIZE = 4;

double time_in_ms() {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

#define CHECK_HIP(cmd)                                                                         \
  do {                                                                                         \
    hipError_t error = cmd;                                                                    \
    if (error != hipSuccess) {                                                                 \
      fprintf(stderr, "error: '%s'(%d) at %s:%d\n", hipGetErrorString(error), error, __FILE__, \
              __LINE__);                                                                       \
      exit(EXIT_FAILURE);                                                                      \
    }                                                                                          \
  } while (0)


// ----------------------------------------- naive -----------------------------------------
__global__ void naive_matrixMultiply(float *A, float *B, float *C, int numARows, int numAColumns,
                                     int numBColumns) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < numARows && col < numBColumns) {
    float sum = 0;
#pragma unroll
    for (int k = 0; k < numAColumns; k++) {
      sum += A[row * numAColumns + k] * B[k * numBColumns + col];
    }
    C[row * numBColumns + col] = sum;
  }
}

__global__ void sgemm_16x16x4(const float *A, const float *B, float *D, int M, int N, int K) {
  int global_index_x = blockIdx.x * blockDim.x + threadIdx.x;
  int global_index_y = blockIdx.y * blockDim.y + threadIdx.y;
  if (global_index_x >= M || global_index_y >= N) {
    return;
  }
  using float4 = __attribute__((__vector_size__(4 * sizeof(float)))) float;
  float4 dmn = {0};

  int mk = threadIdx.y + K * threadIdx.x;
  int kn = threadIdx.x + N * threadIdx.y;
  printf("mk: %d, kn: %d\n", mk, kn);
  printf("A[mk]: %f, B[kn]: %f\n", A[mk], B[kn]);

  float amk = A[mk];
  float bkn = B[kn];
  dmn = __builtin_amdgcn_mfma_f32_16x16x4f32(amk, bkn, dmn, 0, 0, 0);

#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int idx = threadIdx.x + i * N + threadIdx.y * 4 * N;
    printf("idx: %d\n", idx);
    D[idx] = dmn[i];
  }
}

__global__ void sgemm_16x16x4_large(const float *A, const float *B, float *D, int M, int N, int K) {
  int global_index_x = blockIdx.x * blockDim.x + threadIdx.x;
  int global_index_y = blockIdx.y * blockDim.y + threadIdx.y;
  if (global_index_x >= M || global_index_y >= N) {
    return;
  }
  using float4 = __attribute__((__vector_size__(4 * sizeof(float)))) float;
  float4 dmn = {0};

  int mk = threadIdx.y + K * threadIdx.x;
  int kn = threadIdx.x + N * threadIdx.y;
  printf("mk: %d, kn: %d\n", mk, kn);
  printf("A[mk]: %f, B[kn]: %f\n", A[mk], B[kn]);

  // int k_div_4 = K / 4; //8 /4 = 2

  // for (int i = 0; i < k_div_4; i++) {
  //   float amk = A[mk];
  //   float bkn = B[kn];
  //   dmn = dmn + __builtin_amdgcn_mfma_f32_16x16x4f32(amk, bkn, dmn, 0, 0, 0);
  //   mk += 4 * 16;
  //   kn += 4 * 16;
  // }

  float amk = A[mk];
  float bkn = B[kn];

  float4 dmn_1 = {0};
  dmn_1 = __builtin_amdgcn_mfma_f32_16x16x4f32(amk, bkn, dmn_1, 0, 0, 0);

  amk = A[mk + 16 * 4];
  bkn = B[kn + 16 * 4];
  float4 dmn_2 = {0};
  dmn_2 = __builtin_amdgcn_mfma_f32_16x16x4f32(amk, bkn, dmn_2, 0, 0, 0);

  dmn = dmn_1 + dmn_2;

#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int idx = threadIdx.x + i * N + threadIdx.y * 4 * N;
    // printf("idx: %d\n", idx);
    D[idx] = dmn[i];
  }
}

__global__ void sgemm_32x32x2(const float *A, const float *B, float *D, int M, int N, int K) {
  // This kernel computes a 32x32x32 matrix multiplication using a single wavefront.
  using float16 = __attribute__((__vector_size__(16 * sizeof(float)))) float;
  float16 d = {0};  // zero out 16 vanilla VGPRs

  /*
  One invocation of v_mfma_f32_32x32x2f32 accumulates the sum of two outer products,
  two columns of A with two rows of B, into result matrix D (which is in AccVGPRs).
  So we need 16 iterations to compute the full matrix D, starting with the leftmost two
  columns of A and the topmost two colums of B, and then moving two columns to the right
  for A, and down for B, for every iteration.

  For both the two columns of A, and the two rows of B, we use a single regular VGPR.
  With 64 lanes, that covers the 64 values for the two rows/columns of 32 items each.
  For the two A columns: lanes 0-31 cover the 1st column, lanes 32-63 cover the 2nd column.
  For the two B rows: lanes 0-31 cover the 1st row, lanes 32-63 cover the 2nd row.
  Note that A and B are in row-major order.

  This kernel is called with a single wavefront in dim3(32, 2) layout
  */
  int LDA = K;
  int LDB = N;
  int LDD = N;

  int a_idx = LDA * threadIdx.x + threadIdx.y;
  int b_idx = threadIdx.x + LDB * threadIdx.y;

#pragma unroll
  for (int i = 0; i < 16; ++i) {
    const float a = A[a_idx];
    const float b = B[b_idx];

    d = __builtin_amdgcn_mfma_f32_32x32x2f32(a, b, d, 0, 0, 0);
    //                                       ^  ^  ^
    // D(=C)                                  |  |  C(=D)
    //                    two columns of A---|  |--- two rows of B
    a_idx += 2;        // move two columns to the right
    b_idx += 2 * LDB;  // move two rows down
  }

  /*
  Matrix D is a 32 x 32 matrix that is stored in 16 AccVGPRs as follows:
    d[0:3]   cover rows 0-7 (256 floats)
    d[4:7]   cover rows 8-15
    d[8:11]  cover rows 16-23
    d[12:15] cover rows 24-31
  Within each block of 4 AccVGPRs/8 rows (using d[0:3] as an example:
    first 32 lanes of d[0] cover row 0 -  last 32 lanes of d[0] cover row 4
    first 32 lanes of d[1] cover row 1 -  last 32 lanes of d[1] cover row 5
    first 32 lanes of d[2] cover row 2 -  last 32 lanes of d[2] cover row 6
    first 32 lanes of d[3] cover row 3 -  last 32 lanes of d[3] cover row 7
  */

#pragma unroll
  for (int j = 0; j < 4; ++j) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int d_idx = threadIdx.x  // consecutive threads cover 32 consecutive columns
                        + i * LDD    // consecutive registers take consecutive rows of 32 floats
                        + threadIdx.y * 4 * LDD  // last 32 lanes skip 4 rows
                        + j * 2 * 4 * LDD;       // blocks of 4 registers cover 8 rows

      D[d_idx] = d[i + 4 * j];
    }
  }
}



// ----------------------------------------- 10_kernel_warptiling -----------------------------------------

// Define this if not already defined
#define HIP_ASSERT(x) (assert((x) == hipSuccess))
#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))
const int WARPSIZE = 32;  // warpSize is not constexpr

namespace wt {
  template <const int BM, const int BN, const int BK, const int rowStrideA, const int rowStrideB>
  __device__ void loadFromGmem(int N, int K, const float *A, const float *B, float *As, float *Bs,
                               int innerRowA, int innerColA, int innerRowB, int innerColB) {
    for (uint offset = 0; offset + rowStrideA <= BM; offset += rowStrideA) {
      const float4 tmp
          = reinterpret_cast<const float4 *>(&A[(innerRowA + offset) * K + innerColA * 4])[0];
      // float4 tmp;
      // asm("ld.global.nc.v4.f32 {%0, %1, %2, %3}, [%4];"
      //     : "=f"(tmp.x), "=f"(tmp.y), "=f"(tmp.z), "=f"(tmp.w)
      //     : "l"(&A[(innerRowA + offset) * K + innerColA * 4]));
      As[(innerColA * 4 + 0) * BM + innerRowA + offset] = tmp.x;
      As[(innerColA * 4 + 1) * BM + innerRowA + offset] = tmp.y;
      As[(innerColA * 4 + 2) * BM + innerRowA + offset] = tmp.z;
      As[(innerColA * 4 + 3) * BM + innerRowA + offset] = tmp.w;
    }

    for (uint offset = 0; offset + rowStrideB <= BK; offset += rowStrideB) {
      reinterpret_cast<float4 *>(&Bs[(innerRowB + offset) * BN + innerColB * 4])[0]
          = reinterpret_cast<const float4 *>(&B[(innerRowB + offset) * N + innerColB * 4])[0];
      // asm("ld.global.v4.f32 {%0, %1, %2, %3}, [%4];"
      //     : "=f"(Bs[(innerRowB + offset) * BN + innerColB * 4 + 0]),
      //       "=f"(Bs[(innerRowB + offset) * BN + innerColB * 4 + 1]),
      //       "=f"(Bs[(innerRowB + offset) * BN + innerColB * 4 + 2]),
      //       "=f"(Bs[(innerRowB + offset) * BN + innerColB * 4 + 3])
      //     : "l"(&B[(innerRowB + offset) * N + innerColB * 4]));
    }
  }

  template <const int BM, const int BN, const int BK, const int WM, const int WN, const int WMITER,
            const int WNITER, const int WSUBM, const int WSUBN, const int TM, const int TN>
  __device__ void processFromSmem(float *regM, float *regN, float *threadResults, const float *As,
                                  const float *Bs, const uint warpRow, const uint warpCol,
                                  const uint threadRowInWarp, const uint threadColInWarp) {
    for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
      // populate registers for whole warptile
      for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
        for (uint i = 0; i < TM; ++i) {
          regM[wSubRowIdx * TM + i]
              = As[(dotIdx * BM) + warpRow * WM + wSubRowIdx * WSUBM + threadRowInWarp * TM + i];
        }
      }
      for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
        for (uint i = 0; i < TN; ++i) {
          regN[wSubColIdx * TN + i]
              = Bs[(dotIdx * BN) + warpCol * WN + wSubColIdx * WSUBN + threadColInWarp * TN + i];
        }
      }

      // execute warptile matmul
      for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
        for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
          // calculate per-thread results
          for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
            for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
              threadResults[(wSubRowIdx * TM + resIdxM) * (WNITER * TN) + (wSubColIdx * TN)
                            + resIdxN]
                  += regM[wSubRowIdx * TM + resIdxM] * regN[wSubColIdx * TN + resIdxN];
            }
          }
        }
      }
    }
  }
}  // namespace wt

/*
 * @tparam BM The threadblock size for M dimension SMEM caching.
 * @tparam BN The threadblock size for N dimension SMEM caching.
 * @tparam BK The threadblock size for K dimension SMEM caching.
 * @tparam WM M dim of continuous tile computed by each warp
 * @tparam WN N dim of continuous tile computed by each warp
 * @tparam WMITER The number of subwarp tiling steps in M dimension.
 * @tparam WNITER The number of subwarp tiling steps in N dimension.
 * @tparam TM The per-thread tile size for M dimension.
 * @tparam TN The per-thread tile size for N dimension.
 */
template <const int BM, const int BN, const int BK, const int WM, const int WN, const int WNITER,
          const int TM, const int TN, const int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS)
    sgemmWarptiling(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  // Placement of the warp in the threadblock tile
  const uint warpIdx = threadIdx.x / WARPSIZE;  // the warp this thread is in
  const uint warpCol = warpIdx % (BN / WN);
  const uint warpRow = warpIdx / (BN / WN);

  // size of the warp subtile
  constexpr uint WMITER = (WM * WN) / (WARPSIZE * TM * TN * WNITER);
  constexpr uint WSUBM = WM / WMITER;  // 64/2=32
  constexpr uint WSUBN = WN / WNITER;  // 32/2=16

  // Placement of the thread in the warp subtile
  const uint threadIdxInWarp = threadIdx.x % WARPSIZE;          // [0, 31]
  const uint threadColInWarp = threadIdxInWarp % (WSUBN / TN);  // i%(16/4)
  const uint threadRowInWarp = threadIdxInWarp / (WSUBN / TN);  // i/4

  // allocate space for the current blocktile in SMEM
  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  // Move blocktile to beginning of A's row and B's column
  A += cRow * BM * K;
  B += cCol * BN;
  // Move C_ptr to warp's output tile
  C += (cRow * BM + warpRow * WM) * N + cCol * BN + warpCol * WN;

  // calculating the indices that this thread will load into SMEM
  // we'll load 128bit / 32bit = 4 elements per thread at each step
  const uint innerRowA = threadIdx.x / (BK / 4);
  const uint innerColA = threadIdx.x % (BK / 4);
  constexpr uint rowStrideA = (NUM_THREADS * 4) / BK;
  const uint innerRowB = threadIdx.x / (BN / 4);
  const uint innerColB = threadIdx.x % (BN / 4);
  constexpr uint rowStrideB = NUM_THREADS / (BN / 4);

  // allocate thread-local cache for results in registerfile
  float threadResults[WMITER * TM * WNITER * TN] = {0.0};
  // we cache into registers on the warptile level
  float regM[WMITER * TM] = {0.0};
  float regN[WNITER * TN] = {0.0};

  // outer-most loop over block tiles
  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
    wt::loadFromGmem<BM, BN, BK, rowStrideA, rowStrideB>(N, K, A, B, As, Bs, innerRowA, innerColA,
                                                         innerRowB, innerColB);
    __syncthreads();
    wt::processFromSmem<BM, BN, BK, WM, WN, WMITER, WNITER, WSUBM, WSUBN, TM, TN>(
        regM, regN, threadResults, As, Bs, warpRow, warpCol, threadRowInWarp, threadColInWarp);
    A += BK;      // move BK columns to right
    B += BK * N;  // move BK rows down
    __syncthreads();
  }

  // write out the results
  for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
    for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
      // move C pointer to current warp subtile
      float *C_interim = C + (wSubRowIdx * WSUBM) * N + wSubColIdx * WSUBN;
      for (uint resIdxM = 0; resIdxM < TM; resIdxM += 1) {
        for (uint resIdxN = 0; resIdxN < TN; resIdxN += 4) {
          // load C vector into registers
          float4 tmp = reinterpret_cast<float4 *>(
              &C_interim[(threadRowInWarp * TM + resIdxM) * N + threadColInWarp * TN + resIdxN])[0];
          // perform GEMM update in reg
          const int i = (wSubRowIdx * TM + resIdxM) * (WNITER * TN) + wSubColIdx * TN + resIdxN;
          tmp.x = alpha * threadResults[i + 0] + beta * tmp.x;
          tmp.y = alpha * threadResults[i + 1] + beta * tmp.y;
          tmp.z = alpha * threadResults[i + 2] + beta * tmp.z;
          tmp.w = alpha * threadResults[i + 3] + beta * tmp.w;
          // write back
          reinterpret_cast<float4 *>(
              &C_interim[(threadRowInWarp * TM + resIdxM) * N + threadColInWarp * TN + resIdxN])[0]
              = tmp;
        }
      }
    }
  }
}

void hip_sgemm_wt(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
  constexpr int BM = 64;
  constexpr int BN = 64;
  constexpr int BK = 64;
  constexpr int WM = 64;
  constexpr int WN = 64;
  constexpr int WNITER = 2;
  constexpr int TM = 4;
  constexpr int TN = 4;
  constexpr int NUM_THREADS = 256;
  // template <const int BM, const int BN, const int BK, const int WM, const int WN,
  //           const int WNITER, const int TM, const int TN, const int NUM_THREADS>
  sgemmWarptiling<BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS>
      <<<dim3(CEIL_DIV(N, BN), CEIL_DIV(M, BM)), NUM_THREADS>>>(M, N, K, alpha, A, B, beta, C);
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());

  return;
}

// -------------------------------- matmul comp opt -----------------------------------------
template <typename T>
// __global__ void matmul_CompOpt(T *A, T *B, T *C, int M, int K, int N, const int TILE_SIZE, const
// int VECTOR_SIZE) {
__global__ void matmul_CompOpt(T *A, T *B, T *C, int M, int K, int N) {
  /* Computation method optimization.
   * Peform outer product instead of inner product to reduce
   * instructions from shared memory from two to one.
   */
  int bx = blockIdx.x, by = blockIdx.y;
  int tx = threadIdx.x, ty = threadIdx.y;

  // Explicitly allocate As as column-major array
  // to store TILE*TILE submatrix of A.
  __shared__ T As[TILE_SIZE * TILE_SIZE];

  // Allocate register files for sub-result of C at each thread.
  T cv[TILE_SIZE] = {0};

  // Basic iterations is similar with Tiling. But notice that
  // the total number of threads is less than that of Tiling.
  int aBegin = K * TILE_SIZE * by;
  int aEnd = aBegin + K - 1;
  int aStep = TILE_SIZE;

  int bBegin = TILE_SIZE * VECTOR_SIZE * bx;
  int bStep = TILE_SIZE * N;

  for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) {
    // Load Asub with size of TILE*TILE in colomn-major style.
    // Each thread needs to load TILE_SIZE / VECTOR_SIZE values of A.
    int t = VECTOR_SIZE;
    for (int i = 0; i < TILE_SIZE / VECTOR_SIZE; ++i) {
      As[(i * t + ty) + TILE_SIZE * tx] = A[a + K * (i * t + ty) + tx];
    }
    __syncthreads();

    T *ap = As;  // Point to the first address of As, increase later.
    // TODO: global memory ? register ? not clear :(
    T *bp = &B[b + TILE_SIZE * ty + tx];

    for (int i = 0; i < TILE_SIZE; ++i) {
      T bv = *bp;
      // Each thread calculate a vector of C with size of TILE_SIZE.
      for (int j = 0; j < TILE_SIZE; ++j) {
        cv[j] += ap[j] * bv;
      }
      ap += TILE_SIZE;
      bp += N;
    }
    __syncthreads();
  }

  // Store each value of Csub back to C in global memory.
  int c = N * TILE_SIZE * by + TILE_SIZE * VECTOR_SIZE * bx;
  c += TILE_SIZE * ty + tx;
  for (int i = 0; i < TILE_SIZE; ++i) {
    C[c] = cv[i];
    c += N;
  }
}

// -------------------------------------------------- matmul prefetch  -------------------------------------

template <typename T> __global__ void matmul_Prefetch(T *A, T *B, T *C, int M, int K, int N) {
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

// ------------------------------------------ rocmBLAS ------------------------------------------
// rocblas_sgemm(handle, transa, transb, n, m, k, &alpha, db, n, da, k, &beta, dc, n);

// ------------------------------------------ check ------------------------------------------
float *alloc_mat(int R, int C) {
  float *m;
  CHECK_HIP(hipHostMalloc(&m, sizeof(float) * R * C));
  return m;
}

void zero_mat(float *m, int R, int C) { memset(m, 0, sizeof(float) * R * C); }

void check_matmul(float *A, float *B, float *C, int M, int N, int K) {
  printf("Validating...\n");

  float *C_ans = alloc_mat(M, N);
  zero_mat(C_ans, M, N);

#pragma omp parallel for num_threads(20)
  for (int i = 0; i < M; ++i) {
    for (int k = 0; k < K; ++k) {
      for (int j = 0; j < N; ++j) {
        C_ans[i * N + j] += A[i * K + k] * B[k * N + j];
      }
    }
  }

  bool is_valid = true;
  int cnt = 0, thr = 4;
  float eps = 1e-3;
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      float c = C[i * N + j];
      float c_ans = C_ans[i * N + j];
      if (fabsf(c - c_ans) > eps && (c_ans == 0 || fabsf((c - c_ans) / c_ans) > eps)) {
        ++cnt;
        if (cnt <= thr) printf("C[%d][%d] : correct_value = %f, your_value = %f\n", i, j, c_ans, c);
        if (cnt == thr + 1) printf("Too many error, only first %d values are printed.\n", thr);
        is_valid = false;
      }
    }
  }

  if (is_valid) {
    printf("Result: VALID\n");
  } else {
    printf("Result: INVALID\n");
  }
}

int main() {
  // int M = 4096;
  // int K = 4096;
  // int N = 32;


  float alpha = 1.0;
  float beta = 0.0;
  
  printf("------------------------------------------------------------\n");
  printf("M = %d, K = %d, N = %d\n", M, K, N);
  


  rocblas_handle handle;
  rocblas_create_handle(&handle);

  float *A, *B, *D;
  float *d_A, *d_B, *d_D;
  // FILE *f;

  A = (float *)malloc(M * K * sizeof(float));
  B = (float *)malloc(K * N * sizeof(float));
  D = (float *)malloc(M * N * sizeof(float));

  for (int i = 0; i < M * K; ++i) {
    A[i] = 0.1 / (i + 0.001);
  }

  // // save A to bin file
  // FILE *f = fopen("A.bin", "wb");
  // fwrite(A, sizeof(float), M * K, f);
  // fclose(f);

  // //load A from bin file
  // f = fopen("A.bin", "rb");
  // fread(A, sizeof(float), M * K, f);
  // fclose(f);

  for (int i = 0; i < K * N; ++i) {
    B[i] = 0.1 * i;
  }

  // // save B to bin file
  // f = fopen("B.bin", "wb");
  // fwrite(B, sizeof(float), K * N, f);
  // fclose(f);

  // //load B from bin file
  // f = fopen("B.bin", "rb");
  // fread(B, sizeof(float), K * N, f);
  // fclose(f);

  CHECK_HIP(hipMalloc(&d_A, M * K * sizeof(float)));
  CHECK_HIP(hipMalloc(&d_B, K * N * sizeof(float)));
  CHECK_HIP(hipMalloc(&d_D, M * N * sizeof(float)));

  CHECK_HIP(hipMemcpy(d_A, A, M * K * sizeof(float), hipMemcpyHostToDevice));
  CHECK_HIP(hipMemcpy(d_B, B, K * N * sizeof(float), hipMemcpyHostToDevice));



  /*
   * 16X16X4 LARGE KERNEL
   */

  // dim3 block_16(16, 4, 1);
  // dim3 grid_16(1, 1, 1);
  // sgemm_16x16x4_large<<<grid_16, block_16>>>(d_A, d_B, d_D, M, N, K);

  /*
   * 32X32X32 KERNEL
   */

  // dim3 block_32(32, 2, 1);
  // dim3 grid_32(M / 32, N / 32, M*N*K / 32 / 32 / 32);
  // sgemm_32x32x32<<<grid_32, block_32>>>(d_A, d_B, d_D, M, N, K);

  /*
   * COMP OPT KERNEL
   */

  // dim3 threads_comp(TILE_SIZE, VECTOR_SIZE);
  // dim3 grid_comp(N / (TILE_SIZE * VECTOR_SIZE), M / TILE_SIZE);
  // matmul_CompOpt<float><<<grid_comp, threads_comp, 0, 0>>>(d_A, d_B, d_D, M, K, N);


  /*
  * Prefetch KERNEL
  */

  dim3 threads_prefetch(TILE_SIZE, VECTOR_SIZE);
  dim3 grid_prefetch(N / (TILE_SIZE * VECTOR_SIZE), M / TILE_SIZE);
  // dim3 grid_prefetch((N + TILE_SIZE * VECTOR_SIZE - 1) / (TILE_SIZE * VECTOR_SIZE), (M + TILE_SIZE - 1) / TILE_SIZE);
  matmul_Prefetch<float><<<grid_prefetch, threads_prefetch>>>(d_A, d_B, d_D, M, K, N);


  /*
  * ROCBLAS
  */

  // // rocblas_sgemm(handle, transa, transb, n, m, k, &alpha, db, n, da, k, &beta, dc, n);
  rocblas_sgemm(handle, rocblas_operation_none, rocblas_operation_none, N, M, K, &alpha, d_B, N, d_A, K, &beta, d_D, N);

  /*
   * NAIVE KERNEL
   */

  // // Khởi tạo kích thước ma trận
  // int numARows = M; // Số hàng của ma trận A
  // int numAColumns = K; // Số cột của ma trận A
  // int numBColumns = N; // Số cột của ma trận B
  // // Khởi tạo cấu trúc grid và block
  // dim3 threadsPerBlock(16, 16);
  // dim3 blocksPerGrid((numBColumns + threadsPerBlock.x - 1) / threadsPerBlock.x,
  //                     (numARows + threadsPerBlock.y - 1) / threadsPerBlock.y);
  // // Gọi kernel
  // naive_matrixMultiply<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_D, numARows, numAColumns,
  // numBColumns);

  /*
   * CHECK MAIN AND CACULATE TIME
   */
  // double time ;

  // time = time_in_ms();
  // for (int i = 0; i < 10; i++) {
  // hipLaunchKernelGGL(naive_matrixMultiply, dim3(blocksPerGrid), dim3(threadsPerBlock), 0, 0, d_A,
  // d_B, d_D, numARows, numAColumns, numBColumns);
  // }
  // // CHECK_HIP(hipDeviceSynchronize());
  // time = time_in_ms() - time;
  // printf("Time naive_matrixMultiply: %f ms\n", time);

  // // estimate time
  // time = time_in_ms();
  // for (int i = 0; i < 10; i++) {
  //   hipLaunchKernelGGL(sgemm_16x16x4_large, dim3(grid_16), dim3(block_16), 0, 0, d_A, d_B, d_D,
  //   M, N, K);
  // }
  // // CHECK_HIP(hipDeviceSynchronize());
  // time = time_in_ms() - time;
  // printf("Time sgemm_16x16x4_large: %f ms\n", time);

  // time = time_in_ms();
  // for (int i = 0; i < 10; i++) {
  //   hipLaunchKernelGGL(sgemm_32x32x32, dim3(grid_32), dim3(block_32), 0, 0, d_A, d_B, d_D, M, N,
  //   K);
  // }
  // // CHECK_HIP(hipDeviceSynchronize());
  // time = time_in_ms() - time;
  // printf("Time sgemm_32x32x32: %f ms\n", time);

  // hip_sgemm_wt(M, N, K, 1.0, d_A, d_B, 0.0, d_D);
  // hip_sgemm_wt(M, N, K, 1.0, d_A, d_B, 0.0, d_D);


  /*
   * ------------------------------------------------- compare time -------------------------------------------------------------------------
   * Khởi tạo 1 kernel naive trước rocblas_sgemm để cho công bằng
   */

  double time; 
  printf("TILE_SIZE =  %d \n", TILE_SIZE);
  printf("VECTOR_SIZE =  %d \n", VECTOR_SIZE);
  CHECK_HIP(hipMemset(d_D, 0, M * N * sizeof(float)));
  zero_mat(D, M, N);
  CHECK_HIP(hipDeviceSynchronize());



  // time = time_in_ms();
  // for (int i = 0; i < 100; i++) {
  //   matmul_CompOpt<float><<<grid_comp, threads_comp, 0, 0>>>(d_A, d_B, d_D, M, K, N);
  // }
  // CHECK_HIP(hipStreamSynchronize(0));
  // time = time_in_ms() - time;
  // printf("Time matmul_CompOpt: %f ms\n", time);
  // CHECK_HIP(hipMemcpy(D, d_D, M * N * sizeof(float), hipMemcpyDeviceToHost));
  // // CHECK_HIP(hipDeviceSynchronize());
  // check_matmul(A, B, D, M, N, K);
  // // reset d_D and D
  // CHECK_HIP(hipMemset(d_D, 0, M * N * sizeof(float)));
  // zero_mat(D, M, N);





  time = time_in_ms();
  for (int i = 0; i < 100; i++) {
    matmul_Prefetch<float><<<grid_prefetch, threads_prefetch>>>(d_A, d_B, d_D, M, K, N);
  }
  CHECK_HIP(hipStreamSynchronize(0));
  time = time_in_ms() - time;
  printf("Time matmul_Prefetch: %f ms\n", time);
  CHECK_HIP(hipMemcpy(D, d_D, M * N * sizeof(float), hipMemcpyDeviceToHost));
  CHECK_HIP(hipDeviceSynchronize());
  check_matmul(A, B, D, M, N, K);
  // reset d_D and D
  CHECK_HIP(hipMemset(d_D, 0, M * N * sizeof(float)));
  zero_mat(D, M, N);
  


  time = time_in_ms();
  for (int i = 0; i < 100; i++) {
    // rocblas_sgemm(handle, rocblas_operation_none, rocblas_operation_none, N, M, K, &alpha, d_B, N, d_A, K, &beta, d_D, N);

    rocblas_sgemm(handle, rocblas_operation_none, rocblas_operation_none, M, N, K, &alpha, d_A, M, d_B, K, &beta, d_D, M);
  }
  CHECK_HIP(hipStreamSynchronize(0));
  time = time_in_ms() - time;
  printf("Time rocblas_sgemm: %f ms\n", time);
  CHECK_HIP(hipMemcpy(D, d_D, M * N * sizeof(float), hipMemcpyDeviceToHost));
  CHECK_HIP(hipDeviceSynchronize());
  check_matmul(A, B, D, M, N, K);
  // reset d_D and D
  CHECK_HIP(hipMemset(d_D, 0, M * N * sizeof(float)));
  zero_mat(D, M, N);

//----------------------------------------------------------------------------------------

  // CHECK_HIP(hipMemcpy(D, d_D, M * N * sizeof(float), hipMemcpyDeviceToHost));
  // // CHECK_HIP(hipDeviceSynchronize());
  // check_matmul(A, B, D, M, N, K);

}
