#include <hip/hip_ext.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <iostream>
#include <vector>

#include <rocwmma/rocwmma.hpp>

using rocwmma::float16_t;
using rocwmma::float32_t;

#define CHECK_HIP_ERROR(error) \
    if (error != hipSuccess) { \
        std::cerr << "Hip error: " << error << " at file:" << __FILE__ << " line:" << __LINE__ << std::endl; \
        exit(-1); \
    }


// Matrix data initialization
template <typename DataT>
__host__ static inline void fill(DataT* mat, uint32_t m, uint32_t n)
{
    auto ld = n;
    for(int i = 0; i < m; ++i)
    {
        for(int j = 0; j < n; ++j)
        {
             // Generated data
             // Alternate sign every 3 elements
             auto value      = (i * n + j) % 13;
             mat[i * ld + j] = (value % 3) ? -static_cast<DataT>(value) : static_cast<DataT>(value);
        }
    }
}

// Supports BlockM/N square sizes of
// : 16 x 16
// : 32 x 32
const int ROCWMMA_M = 16;
const int ROCWMMA_N = 16;

// Supports ROCWMMA_K sizes as
// : multiples of 16.
const int ROCWMMA_K = 16;

// AMDGCN default wave size
// const int WAVE_SIZE = rocwmma::AMDGCN_WAVE_SIZE;
const int WAVE_SIZE =   rocwmma::Constants::AMDGCN_WAVE_SIZE;

// Thread block
// : T_BLOCK_X must be multiple of WAVE_SIZE.
// Note: Each wave will compute one BLOCK_M x BLOCK_N output block
// Note: Workgroup will compute
//  T_BLOCK_X / WAVE_SIZE x T_BLOCK_Y output blocks
// This thread block will compute (4 x 4 output blocks)
const int T_BLOCK_X = 4 * WAVE_SIZE;
const int T_BLOCK_Y = 4;

// The following device kernel is a naive implementation
// of blocked GEMM. Each wave will compute one BLOCK_M x BLOCK_N
// output block of the M x N x K GEMM, generalized as:
// D = alpha * (A x B) + beta * C
//
// In this simplified example, we assume:
// : A is in row-major format     (m x k)
// : B is in col-major format     (k x n)
// : C, D are in row-major format (m x n)
// : Multiplication is NOT in-place, output is written to D matrix
// : No LDS required
//
// Disclaimer: This is a simplified implementation to demonstrate API usage in
// context of wave-level GEMM computation, and is not optimized.
//
// Launchable device kernel function:
//
__global__ void gemm_wmma_d(uint32_t         m,     // matrix free dim m
                            uint32_t         n,     // matrix free dim n
                            uint32_t         k,     // matrix fixed dim k
                            float16_t const* a,     // device data ptr for matrix A
                            float16_t const* b,     // device data ptr for matrix B
                            float32_t const* c,     // device data ptr for matrix C
                            float32_t*       d,     // device data ptr for matrix D
                            uint32_t         lda,   // leading dimension for matrix A
                            uint32_t         ldb,   // leading dimension for matrix B
                            uint32_t         ldc,   // leading dimension for matrix C
                            uint32_t         ldd,   // leading dimension for matrix D
                            float32_t        alpha, // uniform scalar
                            float32_t        beta)  // uniform scalar
{
    // Create frags with meta-data context for block-wise GEMM decomposition
    // @tp0: fragment context = matrix_a, matrix_b or accumulator
    // @tp1: block size M
    // @tp2: block size N
    // @tp3: block size K
    // @tp4: fragment data type
    // @tp5: data layout = row_major, col_major or void (default)
    auto fragA = rocwmma::fragment<rocwmma::matrix_a, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, float16_t, rocwmma::row_major>();
    auto fragB = rocwmma::fragment<rocwmma::matrix_b, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, float16_t, rocwmma::col_major>();
    auto fragC   = rocwmma::fragment<rocwmma::accumulator, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, float32_t>();
    auto fragAcc = rocwmma::fragment<rocwmma::accumulator, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, float32_t>();

    // Initialize accumulator fragment
    rocwmma::fill_fragment(fragAcc, 0.0f);

     // Tile using a 2D grid
     auto majorWarp = (blockIdx.x * blockDim.x + threadIdx.x) / WAVE_SIZE;
     auto minorWarp = (blockIdx.y * blockDim.y + threadIdx.y);

     // Target C block
     auto cRow = majorWarp * ROCWMMA_M;
     auto cCol = minorWarp * ROCWMMA_N;

    // Bounds check
    if(cRow < m && cCol < n)
    {
         // fragAcc = A x B
         for(int i = 0; i < k; i += ROCWMMA_K)
         {
             // Load the inputs
             rocwmma::load_matrix_sync(fragA, a + (cRow * lda + i), lda);
             rocwmma::load_matrix_sync(fragB, b + (i + cCol * ldb), ldb);

             // Matrix multiply - accumulate using MFMA units
             rocwmma::mma_sync(fragAcc, fragA, fragB, fragAcc);
         }

         // Fetch C matrix
         rocwmma::load_matrix_sync(fragC, c + (cRow * ldc + cCol), ldc, rocwmma::mem_row_major);

         // D = alpha * A x B + beta * C
         for(int i = 0; i < fragC.num_elements; ++i)
         {
             fragC.x[i] = alpha * fragAcc.x[i] + beta * fragC.x[i];
         }

         // Store to D
         rocwmma::store_matrix_sync(d + (cRow * ldd + cCol), fragC, ldd, rocwmma::mem_row_major);
     }
}

// Host side supporting device mgmt and launch code
__host__ void gemm_test(uint32_t m, uint32_t n, uint32_t k, float32_t alpha, float32_t beta)
{
    // Problem size check
    if((m < (ROCWMMA_M * T_BLOCK_X / WAVE_SIZE) || n < (ROCWMMA_N * T_BLOCK_Y) || k < ROCWMMA_K)
        || (m % ROCWMMA_M || n % ROCWMMA_N || k % ROCWMMA_K))
     {
         std::cout << "Unsupported size!\n";
         return;
     }

     int lda = k;
     int ldb = k;
     int ldc = n;
     int ldd = ldc;

     std::cout << "Initializing host data..." << std::endl;

     // Initialize input matrices
     std::vector<float16_t> matrixA(m * k);
     std::vector<float16_t> matrixB(k * n);
     std::vector<float32_t> matrixC(m * n);
     // Fill outputs with NaN to catch contamination
     std::vector<float32_t> matrixD(m * n, std::numeric_limits<float32_t>::signaling_NaN());

     fill(matrixA.data(), m, k);
     fill(matrixB.data(), k, n);
     fill(matrixC.data(), m, n);

     std::cout << "Initializing device data..." << std::endl;

     // Allocate and copy device memory
     float16_t* d_a;
     float16_t* d_b;
     float32_t* d_c;
     float32_t* d_d;

     const size_t bytesA = matrixA.size() * sizeof(float16_t);
     const size_t bytesB = matrixB.size() * sizeof(float16_t);
     const size_t bytesC = matrixC.size() * sizeof(float32_t);
     const size_t bytesD = matrixD.size() * sizeof(float32_t);

     CHECK_HIP_ERROR(hipMalloc(&d_a, bytesA));
     CHECK_HIP_ERROR(hipMalloc(&d_b, bytesB));
     CHECK_HIP_ERROR(hipMalloc(&d_c, bytesC));
     CHECK_HIP_ERROR(hipMalloc(&d_d, bytesD));

     CHECK_HIP_ERROR(hipMemcpy(d_a, matrixA.data(), bytesA, hipMemcpyHostToDevice));
     CHECK_HIP_ERROR(hipMemcpy(d_b, matrixB.data(), bytesB, hipMemcpyHostToDevice));
     CHECK_HIP_ERROR(hipMemcpy(d_c, matrixC.data(), bytesC, hipMemcpyHostToDevice));
     CHECK_HIP_ERROR(hipMemcpy(d_d, matrixD.data(), bytesD, hipMemcpyHostToDevice));

      auto blockDim = dim3(T_BLOCK_X, T_BLOCK_Y);
      auto gridDim  = dim3(rocwmma::ceilDiv(m, ROCWMMA_M * T_BLOCK_X / WAVE_SIZE),
             rocwmma::ceilDiv(n, ROCWMMA_N * T_BLOCK_Y));

      std::cout << "Launching GEMM kernel..." << std::endl;

      hipEvent_t startEvent, stopEvent;
      CHECK_HIP_ERROR(hipEventCreate(&startEvent));
      CHECK_HIP_ERROR(hipEventCreate(&stopEvent));

      hipExtLaunchKernelGGL(gemm_wmma_d,
                       gridDim,
                       blockDim,
                       0, // sharedMemBytes
                       0, // stream
                       startEvent, // Event start
                       stopEvent, // event stop
                       0, // flags
                       m,
                       n,
                       k,
                       d_a,
                       d_b,
                       d_c,
                       d_d,
                       lda,
                       ldb,
                       ldc,
                       ldd,
                       alpha,
                       beta);

      auto elapsedTimeMs = 0.0f;
      CHECK_HIP_ERROR(hipEventSynchronize(stopEvent));
      CHECK_HIP_ERROR(hipEventElapsedTime(&elapsedTimeMs, startEvent, stopEvent));
      CHECK_HIP_ERROR(hipEventDestroy(startEvent));
      CHECK_HIP_ERROR(hipEventDestroy(stopEvent));

      // Release device memory
      CHECK_HIP_ERROR(hipFree(d_a));
      CHECK_HIP_ERROR(hipFree(d_b));
      CHECK_HIP_ERROR(hipFree(d_c));
      CHECK_HIP_ERROR(hipFree(d_d));

      std::cout << "Finished!" << std::endl;
}

int main()
{
    gemm_test(64, 64, 64, 2.1f, 2.1f);
    return 0;
}
