#include "hip_helper.hpp"
#include "thaBLAS.hpp"
#include "utils.hpp"

#include <assert.h>


bool check_mat_mul(float *A, float *B, float *C, int M, int N, int K) {

  float *C_ans;
  alloc_mat(&C_ans, M, N);
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
  int cnt = 0, thr = 10;
  float eps = 1e-3;
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      float c = C[i * N + j];
      float c_ans = C_ans[i * N + j];
      if (fabsf(c - c_ans) > eps &&
          (c_ans == 0 || fabsf((c - c_ans) / c_ans) > eps)) {
        ++cnt;
        if (cnt <= thr)
          printf("C[%d][%d] : correct_value = %f, your_value = %f\n", i, j, c_ans, c);
        if (cnt == thr + 1)
          printf("Too many error, only first %d values are printed.\n", thr);
        is_valid = false;
      }
    }
  }

  // for (int i = 0; i < M; ++i) {
  //   for (int j = 0; j < K; ++j) 
  //     printf("%0.2f ", A[i * K + j]);
  //   printf("\n");
  // }
  // printf("\n");
  // for (int i = 0; i < K; ++i) {
  //   for (int j = 0; j < N; ++j) 
  //     printf("%0.2f ", B[i * N + j]);
  //   printf("\n");
  // }
  // printf("\n"); fflush(stdout);

  if (is_valid) {
    printf("Validation: VALID\n");  fflush(stdout);
    return 1;
  } else {
    printf("Validation: INVALID\n");  fflush(stdout);
    return 0;
  }
}

bool thablas_c2d_Sgemm_test(int M, int N, int K, int num_gpus_to_test)
{
  float *A, *B, *C;
  alloc_mat(&A, M, K);
  alloc_mat(&B, K, N);
  alloc_mat(&C, M, N);
  rand_mat(A, M, K);
  rand_mat(B, K, N);
  zero_mat(C, M, N);

  thablasStatus_t thablasStatus = thablas_c2d_Sgemm(M, N, K, A, B, C, num_gpus_to_test);
  if (thablasStatus != THABLAS_STATUS_SUCCESS)
  {
    printf("Validation: THABLAS error\n");  fflush(stdout);
    return 0;
  }

  return check_mat_mul(A, B, C, M, N, K);
}

bool thablas_c2d_Svds_test(int n, int num_gpus_to_test)
{
  float *A, *B;
  alloc_vec(&A, n);
  alloc_vec(&B, n);
  rand_vec(A, n);
  rand_vec(B, n);
  float val = (float)rand() / (float)RAND_MAX;

  thablasStatus_t thablasStatus = thablas_c2d_Svds(n, A, B, val, num_gpus_to_test);
  if (thablasStatus != THABLAS_STATUS_SUCCESS)
      return 0;

  float *B_ans;
  alloc_vec(&B_ans, n);
  zero_vec(B_ans, n);

  #pragma omp parallel for num_threads(20)
  for (int i = 0; i < n; ++i) {
    B_ans[i] = A[i] / val;
  }
    
  bool is_valid = true;
  int cnt = 0, thr = 10;
  float eps = 1e-3;
  for (int i = 0; i < n; ++i) {
    float b = B[i];
    float b_ans = B_ans[i];
    if (fabsf(b - b_ans) > eps &&
        (b_ans == 0 || fabsf((b - b_ans) / b_ans) > eps)) {
      ++cnt;
      if (cnt <= thr)
        printf("B[%d] : correct_value = %f, your_value = %f\n", i, b_ans, b);
      if (cnt == thr + 1)
        printf("Too many error, only first %d values are printed.\n", thr);
      is_valid = false;
    }
  }

  if (is_valid) {
    printf("Validation: VALID\n");
    return 1;
  } else {
    printf("Validation: INVALID\n");
    return 0;
  }
}



int main()
{
  bool all_valid = 1;

  all_valid = std::min(all_valid, thablas_c2d_Sgemm_test(3, 3, 3, 2));
  assert(all_valid);
  all_valid = std::min(all_valid, thablas_c2d_Sgemm_test(100, 100, 100, 3));
  assert(all_valid);
  all_valid = std::min(all_valid, thablas_c2d_Sgemm_test(1000, 1000, 1000, 4));
  assert(all_valid);
  printf("GEMM PASSED\n");

  all_valid = std::min(all_valid, thablas_c2d_Svds_test(10, 2));
  assert(all_valid);
  all_valid = std::min(all_valid, thablas_c2d_Svds_test(1000, 3));
  assert(all_valid);
  all_valid = std::min(all_valid, thablas_c2d_Svds_test(100000, 4));
  assert(all_valid);
  printf("VDS PASSED\n");

  return 0;
}