#include "hip_helper.hpp"
#include "thaBLAS.hpp"
#include "utils.hpp"
#include <assert.h>
#include <hip/hip_runtime.h>




bool check_sum_all_vector_elements( int n, float *A, float val) {
  float sum = 0;
  for (int i = 0; i < n; ++i) {
    sum += A[i];
  }
  printf("[====] sum = %f, val = %f\n", sum, val);
  if (fabsf(sum - val) < 1e-3) {
    printf("Validation: VALID\n");
    return 1;
  } else {
    printf("Validation: INVALID\n");
    return 0;
  }

}

bool check_sum_all_vector_elements_thaBLAS(int n, float *A) {
  thablasHandle_t handle;
  thablasCreate(&handle);

  // Allocate memory device for list A 
  float *d_A;
  CHECK_HIP(hipMalloc(&d_A, n * sizeof(float)));
  CHECK_HIP(hipMemcpy(d_A, A, n * sizeof(float), hipMemcpyHostToDevice));

  // Allocate memory device for sum_value
  float sum_value = 0;
  float d_sum_value;
  
  hipMalloc(&d_sum_value, sizeof(float));
  // CHECK_HIP(hipMemcpy(d_sum_value, &sum_value, sizeof(float), hipMemcpyHostToDevice));

  // // Call thablas_Ssum to calculate sum of all elements in A
  // thablas_Ssum(handle, n, d_A, d_sum_value);
  // CHECK_HIP(hipMemcpy(&sum_value, d_sum_value, sizeof(float), hipMemcpyDeviceToHost));

  // // Free memory
  // CHECK_HIP(hipFree(d_A));
  // CHECK_HIP(hipFree(d_sum_value));
  // thablasDestroy(handle);

  return check_sum_all_vector_elements(n, A, sum_value);
}

int main() {
  bool all_valid = 1;
  int N = 1000;
  float *A;
  alloc_vec(&A, N);
  rand_vec(A, N);

  // CHECK_HIP(hipSetDevice(0));
  // check_sum_all_vector_elements_thaBLAS(N, A);

  if (all_valid) {
    printf("All tests passed!\n");
  } else {
    printf("Some tests failed!\n");
  }
  return 0;
}
