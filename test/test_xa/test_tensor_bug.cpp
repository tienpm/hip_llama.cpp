#include <hip/hip_ext.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <iostream>
#include <stdexcept>
#include <vector>

#include <rocwmma/rocwmma.hpp>

using rocwmma::float32_t;
using rocwmma::float32_t;
// define CHECK_HIP_ERROR
#define CHECK_HIP_ERROR(cmd)                                                                        \
    do                                                                                              \
    {                                                                                               \
        hipError_t error = cmd;                                                                     \
        if (error != hipSuccess)                                                                    \
        {                                                                                           \
            std::cerr << "Hip Error: " << hipGetErrorString(error) << " at line " << __LINE__       \
                      << std::endl;                                                                  \
            exit(error);                                                                            \
        }                                                                                           \
    } while (0)

void matmul(float32_t* hA, float32_t* hB, float32_t* hC, int M, int N, int K)
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
            //  auto value      = (i * n + j) % 13;
            //  mat[i * ld + j] = (value % 3) ? -stat
            // ic_cast<DataT>(value) : static_cast<DataT>(value);

            // random value from 0 to 1
            mat[i * ld + j] = static_cast<DataT>(rand()) / static_cast<DataT>(RAND_MAX)*0.1;
        }
    }
}

// Matrix data initialization
template <typename DataT>
__host__ static inline void fill_C(DataT* mat, uint32_t m, uint32_t n)
{
    auto ld = n;
    for(int i = 0; i < m; ++i)
    {
        for(int j = 0; j < n; ++j)
        {
             // Generated data
             // Alternate sign every 3 elements
            //  auto value      = (i * n + j) % 13;
            //  mat[i * ld + j] = (value % 3) ? -static_cast<DataT>(value) : static_cast<DataT>(value);
            mat[i * ld + j] = 10;
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
const int WAVE_SIZE = 64;

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
                            float32_t const* a,     // device data ptr for matrix A
                            float32_t const* b,     // device data ptr for matrix B
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
    auto fragA = rocwmma::fragment<rocwmma::matrix_a, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, float32_t, rocwmma::row_major>();
    auto fragB = rocwmma::fragment<rocwmma::matrix_b, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, float32_t, rocwmma::col_major>();
    // auto fragC   = rocwmma::fragment<rocwmma::accumulator, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, float32_t>();
    auto fragAcc = rocwmma::fragment<rocwmma::accumulator, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, float32_t>();

    // Initialize accumulator fragment
    rocwmma::fill_fragment(fragAcc, float32_t(0.0) );

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

        //  // Fetch C matrix
        //  rocwmma::load_matrix_sync(fragC, c + (cRow * ldc + cCol), ldc, rocwmma::mem_row_major);

        //  // D = alpha * A x B + beta * C
        //  for(int i = 0; i < fragC.num_elements; ++i)
        //  {
        //      fragC.x[i] = alpha * fragAcc.x[i] + beta * fragC.x[i];
        //  }

         // Store to D
         rocwmma::store_matrix_sync(d + (cRow * ldd + cCol), fragAcc, ldd, rocwmma::mem_row_major);
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
     std::vector<float32_t> matrixA(m * k);
     std::vector<float32_t> matrixB(k * n);
     std::vector<float32_t> matrixC(m * n);
     // Fill outputs with NaN to catch contamination
     std::vector<float32_t> matrixD(m * n, std::numeric_limits<float32_t>::signaling_NaN());

     fill<float32_t>(matrixA.data(), m, k);
     fill<float32_t>(matrixB.data(), k, n);
     fill<float32_t>(matrixC.data(), m, n);

     std::cout << "Initializing device data..." << std::endl;

    // // ép kiểu từ float sang float32_t
    // for (int i = 0; i < matrixA.size(); i++) {
    //     matrixA[i] = (float32_t)matrixA[i];
    // }
    // for (int i = 0; i < matrixB.size(); i++) {
    //     matrixB[i] = (float32_t)matrixB[i];
    // }
    


     // Allocate and copy device memory
     float32_t* d_a;
     float32_t* d_b;
     float32_t* d_c;
     float32_t* d_d;

     const size_t bytesA = matrixA.size() * sizeof(float32_t);
     const size_t bytesB = matrixB.size() * sizeof(float32_t);
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
      std::cout << "Finished!" << std::endl;

    //   // Release device memory
    //   CHECK_HIP_ERROR(hipFree(d_a));
    //   CHECK_HIP_ERROR(hipFree(d_b));
    //   CHECK_HIP_ERROR(hipFree(d_c));
    //   CHECK_HIP_ERROR(hipFree(d_d));

    // copy output from device to host
    float * out_gpu = new float32_t[m * n];
    CHECK_HIP_ERROR(hipMemcpy(out_gpu, d_d, m * n * sizeof(float32_t), hipMemcpyDeviceToHost));

    // // compare with CPU
    // float * out_cpu = new float[m * n];
    // // cpu_A: ép kiểu từ float16_t sang float
    // std::vector<float> cpu_A(matrixA.size());
    // std::vector<float> cpu_B(matrixB.size());

    // for (int i = 0; i < matrixA.size(); i++) {
    //     cpu_A[i] = (float)matrixA[i];
    // }
    // for (int i = 0; i < matrixB.size(); i++) {
    //     cpu_B[i] = (float)matrixB[i];
    // }
 
    //ouput_cpu 
    float32_t * out_cpu = new float32_t[m * n];


    matmul(matrixA.data(), matrixB.data(), out_cpu, m, n, k);

    int errors = 0;
    int max_errors = 150;
    float espilon = 0.0001;
    for (int i = 0; i < m * n; i++) {
        if (abs(out_gpu[i] - out_cpu[i]) > espilon) {
            errors++;
            if (errors < max_errors) {
                std::cout << "Error at " << i << " : " << out_gpu[i] << " != " << out_cpu[i] << std::endl;
            }
        }
    }

    if (errors > 0) {
        std::cout << "ERROR " << errors << std::endl;
    } else {
        std::cout << "PASS" << std::endl;
    }

}

int main()
{
    gemm_test(256, 256, 256, 1.0f, 0.0f);
    return 0;
}
