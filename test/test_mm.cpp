#include <hip/hip_runtime.h>

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>

// include file test_tensor.cpp
// #include "test_tensor.cpp"

// CHECK_HIP
#define CHECK_HIP(cmd)                                                          \
  do {                                                                         \
    hipError_t e = cmd;                                                        \
    if (e != hipSuccess) {                                                     \
      fprintf(stderr, "Failed: %s\nerror: '%s'(%d) at %s:%d\n", #cmd,          \
              hipGetErrorString(e), e, __FILE__, __LINE__);                    \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

// Define this if not already defined
#ifndef HIP_ASSERT
#  define HIP_ASSERT(x) (assert((x) == hipSuccess))
#endif

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
} // namespace wt

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
template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WNITER, const int TM, const int TN, const int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS)
    sgemmWarptiling(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  // Placement of the warp in the threadblock tile
  const uint warpIdx = threadIdx.x / WARPSIZE; // the warp this thread is in
  const uint warpCol = warpIdx % (BN / WN);
  const uint warpRow = warpIdx / (BN / WN);

  // size of the warp subtile
  constexpr uint WMITER = (WM * WN) / (WARPSIZE * TM * TN * WNITER);
  constexpr uint WSUBM = WM / WMITER; // 64/2=32
  constexpr uint WSUBN = WN / WNITER; // 32/2=16

  // Placement of the thread in the warp subtile
  const uint threadIdxInWarp = threadIdx.x % WARPSIZE;         // [0, 31]
  const uint threadColInWarp = threadIdxInWarp % (WSUBN / TN); // i%(16/4)
  const uint threadRowInWarp = threadIdxInWarp / (WSUBN / TN); // i/4

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
    wt::loadFromGmem<BM, BN, BK, rowStrideA, rowStrideB>(
        N, K, A, B, As, Bs, innerRowA, innerColA, innerRowB, innerColB);
    __syncthreads();
    wt::processFromSmem<BM, BN, BK, WM, WN, WMITER, WNITER, WSUBM, WSUBN, TM,
                        TN>(regM, regN, threadResults, As, Bs, warpRow, warpCol,
                            threadRowInWarp, threadColInWarp);
    A += BK;     // move BK columns to right
    B += BK * N; // move BK rows down
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
              &C_interim[(threadRowInWarp * TM + resIdxM) * N +
                         threadColInWarp * TN + resIdxN])[0];
          // perform GEMM update in reg
          const int i = (wSubRowIdx * TM + resIdxM) * (WNITER * TN) +
                        wSubColIdx * TN + resIdxN;
          tmp.x = alpha * threadResults[i + 0] + beta * tmp.x;
          tmp.y = alpha * threadResults[i + 1] + beta * tmp.y;
          tmp.z = alpha * threadResults[i + 2] + beta * tmp.z;
          tmp.w = alpha * threadResults[i + 3] + beta * tmp.w;
          // write back
          reinterpret_cast<float4 *>(
              &C_interim[(threadRowInWarp * TM + resIdxM) * N +
                         threadColInWarp * TN + resIdxN])[0] = tmp;
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


void matmul(float* hA, float* hB, float* hC, int M, int N, int K)
         {
    for (int i = 0; i < M; i++) {
      for (int j = 0; j < N; j++) {
        float sum = 0;
        for (int k = 0; k < K; k++) {
          sum += hA[i * K + k] * hB[k * N + j];
        }
        hC[i * N + j] = sum;
      }
    }
}

int main() {
  int M = 64;
  int N = 64;
  int K = 64;
  float alpha = 1.0;
  float beta = 0.0;
  float *A, *B, *C;
  CHECK_HIP(hipMalloc(&A, M * K * sizeof(float)));
  CHECK_HIP(hipMalloc(&B, K * N * sizeof(float)));
  CHECK_HIP(hipMalloc(&C, M * N * sizeof(float)));
  // Initialize A and B
  float *hA = (float *)malloc(M * K * sizeof(float));
  float *hB = (float *)malloc(K * N * sizeof(float));
  for (int i = 0; i < M * K; i++) {
    // random value
    hA[i] = (rand() % 100) / 100.0;
  }
  for (int i = 0; i < K * N; i++) {
    // random value
    hB[i] = (rand() % 100) / 100.0;

  }

  CHECK_HIP(hipMemcpy(A, hA, M * K * sizeof(float), hipMemcpyHostToDevice));
  CHECK_HIP(hipMemcpy(B, hB, K * N * sizeof(float), hipMemcpyHostToDevice));



  hip_sgemm_wt(M, N, K, alpha, A, B, beta, C);

  float *hC = (float *)malloc(M * N * sizeof(float));
  CHECK_HIP(hipMemcpy(hC, C, M * N * sizeof(float), hipMemcpyDeviceToHost));

  float *hC_ref = (float *)malloc(M * N * sizeof(float));
  matmul(hA, hB, hC_ref, M, N, K);

  int errors = 0;
  int errors_max = 10;
  float epsilon = 1e-5;
  for (int i = 0; i < M * N; i++) {
    if (fabs(hC[i] - hC_ref[i]) > epsilon) {
      errors++;
      if (errors < errors_max) {
        printf("hC[%d] = %f, hC_ref[%d] = %f\n", i, hC[i], i, hC_ref[i]);
      }
    }
  }
  printf("PASSED\n");

  free(hA);
  free(hB);
  free(hC);
  CHECK_HIP(hipFree(A));
  CHECK_HIP(hipFree(B));
  CHECK_HIP(hipFree(C));
  return 0;
}
