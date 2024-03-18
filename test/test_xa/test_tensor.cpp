// #include <hip/hip_ext.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <iostream>
#include <vector>

#include <rocwmma/rocwmma.hpp>

using rocwmma::float16_t;
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


// Matrix data initialization
template <typename DataT>
__host__ static inline void fill(DataT* mat, int m, int n)
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
            mat[i * ld + j] = 0.001*( (i + ld ) % 3 );
        }
    }
}

// Matrix data initialization
template <typename DataT>
__host__ static inline void fill_C(DataT* mat, int m, int n)
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
            mat[i * ld + j] = 100;
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
__global__ void gemm_wmma_d(int         m,     // matrix free dim m
                            int         n,     // matrix free dim n
                            int         k,     // matrix fixed dim k
                            float * a,     // device data ptr for matrix A
                            float * b,     // device data ptr for matrix B
                            float * c,     // device data ptr for matrix C
                            float *       d,     // device data ptr for matrix D
                            int         lda,   // leading dimension for matrix A
                            int         ldb,   // leading dimension for matrix B
                            int         ldc,   // leading dimension for matrix C
                            int         ldd,   // leading dimension for matrix D
                            float        alpha, // uniform scalar
                            float        beta)  // uniform scalar
{
    // Create frags with meta-data context for block-wise GEMM decomposition
    // @tp0: fragment context = matrix_a, matrix_b or accumulator
    // @tp1: block size M
    // @tp2: block size N
    // @tp3: block size K
    // @tp4: fragment data type
    // @tp5: data layout = row_major, col_major or void (default)

    // ép kiểu dữ liệu cho các fragment, ép kiểu a, b về float16_t, ép kiểu c, d về float32_t
    float16_t * a_fp16 = (float16_t*)a;
    float16_t * b_fp16 = (float16_t*)b;
    float * c_fp32 = (float*)c;
    float * d_fp32 = (float*)d;
    // float32_t * c_fp32 = (float32_t*)c;
    // float32_t * d_fp32 = (float32_t*)d;
    // Create fragments
    
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
             rocwmma::load_matrix_sync(fragA, a_fp16 + (cRow * lda + i), lda);
             rocwmma::load_matrix_sync(fragB, b_fp16 + (i + cCol * ldb), ldb);

             // Matrix multiply - accumulate using MFMA units
             rocwmma::mma_sync(fragAcc, fragA, fragB, fragAcc);
         }

         // Fetch C matrix
         rocwmma::load_matrix_sync(fragC, c_fp32 + (cRow * ldc + cCol), ldc, rocwmma::mem_row_major);

         // D = alpha * A x B + beta * C
         for(int i = 0; i < fragC.num_elements; ++i)
         {
             fragC.x[i] = alpha * fragAcc.x[i] + beta * fragC.x[i];
         }

         // Store to D
         rocwmma::store_matrix_sync(d_fp32 + (cRow * ldd + cCol), fragC, ldd, rocwmma::mem_row_major);

         // Synchronize to make sure the result is visible
            __syncthreads();
    }

     // ép kiểu d về float 
    d = static_cast<float *>(d_fp32);
}

// Host side supporting device mgmt and launch code
__host__ void gemm_test(int m, int n, int k, float alpha, float beta)
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

    float *hA, *hB, *hC, *hD;
    float *d_a, *d_b, *d_c, *d_d;

    // Allocate host memory
    hA = (float*)malloc(m * k * sizeof(float));
    hB = (float*)malloc(k * n * sizeof(float));
    hC = (float*)malloc(m * n * sizeof(float));
    hD = (float*)malloc(m * n * sizeof(float));

    // Allocate device memory
    CHECK_HIP_ERROR(hipMalloc(&d_a, m * k * sizeof(float)));
    CHECK_HIP_ERROR(hipMalloc(&d_b, k * n * sizeof(float)));
    CHECK_HIP_ERROR(hipMalloc(&d_c, m * n * sizeof(float)));
    CHECK_HIP_ERROR(hipMalloc(&d_d, m * n * sizeof(float)));

    // Initialize host data
    fill<float>(hA, m, k);
    fill<float>(hB, k, n);
    fill_C<float>(hC, m, n);

    // Copy data to device
    CHECK_HIP_ERROR(hipMemcpy(d_a, hA, m * k * sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_b, hB, k * n * sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_c, hC, m * n * sizeof(float), hipMemcpyHostToDevice));



    // Launch kernel

    auto blockDim = dim3(T_BLOCK_X, T_BLOCK_Y);
    auto gridDim  = dim3(rocwmma::ceilDiv(m, ROCWMMA_M * T_BLOCK_X / WAVE_SIZE),
            rocwmma::ceilDiv(n, ROCWMMA_N * T_BLOCK_Y));
   
    gemm_wmma_d<<<gridDim, blockDim>>>(m, n, k, d_a, d_b, d_c, d_d, lda, ldb, ldc, ldd, alpha, beta);

    // Copy data back to host
    CHECK_HIP_ERROR(hipMemcpy(hD, d_d, m * n * sizeof(float), hipMemcpyDeviceToHost));
    CHECK_HIP_ERROR(hipDeviceSynchronize());

    // Validate results

    // Compute reference
    float* hRef = (float*)malloc(m * n * sizeof(float));
    matmul(hA, hB, hRef, m, n, k);

    // Compare results
    int errors = 0;
    int maxErrors = 10;

    for(int i = 0; i < m * n; i++)
    {
        if(hRef[i] != hD[i])
        {
            if(errors < maxErrors)
            {
                std::cout << "Mismatch at " << i << " expected " << hRef[i] << " actual " << hD[i] << std::endl;
            }
            errors++;
        }
    }

    if (errors > 0) {
        std::cout << "Total errors: " << errors << std::endl;
    } else {
        std::cout << "Validation passed!" << std::endl;
    }
}
int main()
{
    gemm_test(256, 256, 256, 1.0f, 0.0f);
    return 0;
}
