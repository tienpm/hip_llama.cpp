#include "hip_helper.hpp"
#include "thaBLAS.hpp"
#include "thaDNN.hpp"
#include "utils.hpp"

#include <assert.h>

// copied from original CPU code
void cpu_rmsnorm(float* o, float* x, float* weight, int size) {
  // calculate sum of squares
  float ss = 0.0f;
  for (int j = 0; j < size; j++) {
    ss += x[j] * x[j];
  }
  ss /= size;

  ss += 1e-5f;
  ss = 1.0f / sqrtf(ss);
  // normalize and scale
  for (int j = 0; j < size; j++) {
    o[j] = weight[j] * (ss * x[j]);
  }
}

bool test_thaDNN_h2d_s_rmsnorm(int size)
{
  float *o, *x, *weight;
  alloc_vec(&o, size);
  alloc_vec(&x, size);
  alloc_vec(&weight, size);
  rand_vec(x, size);
  rand_vec(weight, size);
  zero_vec(o, size);

  thablasStatus_t thablasStatus = thaDNN_h2d_s_rmsnorm(o, x, weight, size);
  if (thablasStatus != THABLAS_STATUS_SUCCESS)
      return 0;

  float *o_h;
  alloc_vec(&o_h, size);
  zero_vec(o_h, size);
  cpu_rmsnorm(o_h, x, weight, size);  

  bool is_valid = true;
  int cnt = 0, thr = 10;
  float eps = 1e-3;
  for (int i = 0; i < size; ++i) {
    float o_gpu = o[i];
    float o_ans = o_h[i];
    if (fabsf(o_gpu - o_ans) > eps &&
        (o_ans == 0 || fabsf((o_gpu - o_ans) / o_ans) > eps)) {
      ++cnt;
      if (cnt <= thr)
        printf("O[%d] : correct_value = %f, your_value = %f\n", i, o_ans, o_gpu);
      if (cnt == thr + 1)
        printf("Too many error, only first %d values are printed.\n", thr);
      is_valid = false;
    }
  }

  if (is_valid) {
    printf("Validation: VALID\n"); fflush(stdout);
    return 1;
  } else {
    printf("Validation: INVALID\n"); fflush(stdout);
    return 0;
  }
}



int main()
{
  bool all_valid = 1;

  all_valid = std::min(all_valid, test_thaDNN_h2d_s_rmsnorm(768)); // 768 only
  assert(all_valid);
  all_valid = std::min(all_valid, test_thaDNN_h2d_s_rmsnorm(768)); // 768 only
  assert(all_valid);
  all_valid = std::min(all_valid, test_thaDNN_h2d_s_rmsnorm(768)); // 768 only
  assert(all_valid);
  printf("RMSNORM PASSED\n");

  return 0;
}