/* ************************************************************************
 * Copyright (C) 2016-2023 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
 * ies of the Software, and to permit persons to whom the Software is furnished
 * to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
 * PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
 * CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * ************************************************************************ */

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <cstdlib>
#include <iostream>
using namespace std;

#define CHECK_HIP_ERROR(cmd)                                                         \
    do                                                                                \
    {                                                                                 \
        hipError_t error = cmd;                                                       \
        if(error != hipSuccess)                                                       \
        {                                                                             \
            fprintf(stderr, "error: '%s'(%d) at %s:%d\n", hipGetErrorString(error),   \
                    error, __FILE__, __LINE__);                                       \
            exit(EXIT_FAILURE);                                                       \
        }                                                                             \
    } while(0)


// #define DIM1 4
// #define DIM2 4
// #define DIM3 2

template <typename T>
void mat_mat_mult(T        alpha,
                  T        beta,
                  int      M,
                  int      N,
                  int      K,
                  const T* A,
                  int      As1,
                  int      As2,
                  const T* B,
                  int      Bs1,
                  int      Bs2,
                  T*       C,
                  int      Cs1,
                  int      Cs2)
    // mat_mat_mult<float>(alpha,
    //                     beta,
    //                     m,
    //                     n,
    //                     k,
    //                     ha.data(),
    //                     a_stride_1,
    //                     a_stride_2,
    //                     hb.data(),
    //                     b_stride_1,
    //                     b_stride_2,
    //                     hc_gold.data(),
    //                     1,
    //                     ldc);
{
    printf("--------------------------------------------------\n");
    printf("M, N, K = %d, %d, %d\n", M, N, K);
    printf("As1, As2 = %d, %d\n", As1, As2);
    printf("Bs1, Bs2 = %d, %d\n", Bs1, Bs2);
    printf("Cs1, Cs2 = %d, %d\n", Cs1, Cs2);
    for(int i1 = 0; i1 < M; i1++)
    {
        for(int i2 = 0; i2 < N; i2++)
        {
            T t = 0.0;
            for(int i3 = 0; i3 < K; i3++)
            {
                // printf("i1, i2, i3 = %d, %d, %d\n", i1, i2, i3);
                // printf(" i1 * As1 + i3 * As2 = %d, i3 * Bs1 + i2 * Bs2 = %d\n", i1 * As1 + i3 * As2, i3 * Bs1 + i2 * Bs2);
                t += A[i1 * As1 + i3 * As2] * B[i3 * Bs1 + i2 * Bs2];
            }
            C[i1 * Cs1 + i2 * Cs2] =  t;
        }
    }
}


// -------------------------------------------------- check --------------------------------------------------
float *alloc_mat(int R, int C) {
  float *m = (float *)malloc(sizeof(float) * R * C);
  return m;
}

void zero_mat(float *m, int R, int C) {
  memset(m, 0, sizeof(float) * R * C);
}

void check_matmul(float *A, float *B, float *C, int M, int N, int K) {
  printf("Validating...\n");

  float *C_ans = alloc_mat(M, N);
  zero_mat(C_ans, M, N);

#pragma omp parallel for num_threads(20)
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      for (int k = 0; k < K; ++k) {
        C_ans[i * N + j] += A[i * K + k] * B[k * N + j];
      }
    }
  }

  bool is_valid = true;
  int cnt = 0, thr = 10;
  float eps = 1e-5;
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      float c = C[i * N + j];
      float c_ans = C_ans[i * N + j];
      if (fabsf(c - c_ans) > eps &&
          (c_ans == 0 || fabsf((c - c_ans) / c_ans) > eps)) {
        ++cnt;
        if (cnt <= thr)
          printf("C[%d][%d] : correct_value = %f, your_value = %f\n", i, j,
                 c_ans, c);
        if (cnt == thr + 1)
          printf("Too many error, only first %d values are printed.\n", thr);
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

int main()
{
    rocblas_operation transa = rocblas_operation_none, transb = rocblas_operation_none;
    float             alpha = 1.0, beta = 0.0;

    int m = 3000;
    int n = 1000;
    int k = 8;
    int lda, ldb, ldc, size_a, size_b, size_c;
    int         a_stride_1, a_stride_2, b_stride_1, b_stride_2;
    cout << "sgemm example" << std::endl;
    {
        // lda        = m;
        lda = n;
        size_a     = k * m;
        a_stride_1 = 1;
        a_stride_2 = lda;
        cout << "N";
    }

    {
        // ldb        = k;
        ldb = k;
        size_b     = n * k;
        b_stride_1 = 1;
        b_stride_2 = ldb;
        cout << "N: ";
    }
   
    // ldc    = m;
    ldc = n
    size_c = n * m;

    // Naming: da is in GPU (device) memory. ha is in CPU (host) memory
    float *ha, *hb, *hc, *hc_gold;
    ha = (float*)malloc(size_a * sizeof(float));
    hb = (float*)malloc(size_b * sizeof(float));
    hc = (float*)malloc(size_c * sizeof(float));
    hc_gold = (float*)malloc(size_c * sizeof(float));
    // initial data on host
    srand(1);
    for(int i = 0; i < size_a; ++i)
    {
        // ha[i] = rand() % 17;
        ha[i] = i;
    }
    for(int i = 0; i < size_b; ++i)
    {
        // hb[i] = rand() % 17;
        // random from 0 to 1
        hb[i] = i * 10;
    }

    for(int i = 0; i < size_c; ++i)
    {
        hc[i] = -1000;
    }

    // allocate memory on device
    float *da, *db, *dc;
    CHECK_HIP_ERROR(hipMalloc(&da, size_a * sizeof(float)));
    CHECK_HIP_ERROR(hipMalloc(&db, size_b * sizeof(float)));
    CHECK_HIP_ERROR(hipMalloc(&dc, size_c * sizeof(float)));

    // copy matrices from host to device
    CHECK_HIP_ERROR(hipMemcpy(da, ha, sizeof(float) * size_a, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(db, hb, sizeof(float) * size_b, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dc, hc, sizeof(float) * size_c, hipMemcpyHostToDevice));

    rocblas_handle handle;
    rocblas_create_handle(&handle);
    printf("lda, ldb, ldc = %d, %d, %d\n", lda, ldb, ldc);
    // rocblas_sgemm(handle, transa, transb, m, n, k, &alpha, da, lda, db, ldb, &beta, dc, ldc);
    rocblas_sgemm(handle, transa, transb, n, m, k, &alpha, db, n, da, k, &beta, dc, n);

    // copy output from device to CPU
    CHECK_HIP_ERROR(hipMemcpy(hc, dc, sizeof(float) * size_c, hipMemcpyDeviceToHost));

    cout << "m, n, k, lda, ldb, ldc = " << m << ", " << n << ", " << k << ", " << lda
                 << ", " << ldb << ", " << ldc << std::endl;

    float max_relative_error = std::numeric_limits<float>::min();

// --------------------------- CHECK MATMUL ORIGIN --------------------------------------------------------------
    // calculate golden or correct result
    mat_mat_mult<float>(alpha,
                        beta,
                        m,
                        n,
                        k,
                        ha,
                        a_stride_1,
                        a_stride_2,
                        hb,
                        b_stride_1,
                        b_stride_2,
                        hc_gold,
                        1,
                        ldc);

    for(int i = 0; i < size_c; i++)
    {
        float relative_error = (hc_gold[i] - hc[i]) / hc_gold[i];
        relative_error       = relative_error > 0 ? relative_error : -relative_error;
        max_relative_error
            = relative_error < max_relative_error ? max_relative_error : relative_error;
    }
    float eps       = std::numeric_limits<float>::epsilon();
    float tolerance = 10;
    if(max_relative_error != max_relative_error || max_relative_error > eps * tolerance)
    {
        cout << "FAIL: max_relative_error = " << max_relative_error << std::endl;
    }
    else
    {
        cout << "PASS: max_relative_error = " << max_relative_error << std::endl;
    }

// -------------------------------- CHECK MATMUL SNU --------------------------------------------------------------
    check_matmul(ha, hb, hc, m, n, k);

// ----------------------------------------------------------------------------------------------------------------
    CHECK_HIP_ERROR(hipFree(da));
    CHECK_HIP_ERROR(hipFree(db));
    CHECK_HIP_ERROR(hipFree(dc));
    rocblas_destroy_handle(handle);
    return EXIT_SUCCESS;
}
