#include "hip_helper.hpp"
#include "thaBLAS.hpp"
#include "thaDNN.hpp"
#include "seq.hpp"

#include <assert.h>
#include <chrono>

bool test_thaDNN_h2d_s_rmsnorm(int size)
{
  float *o, *x, *weight;
  alloc_vec(&o, size);
  alloc_vec(&x, size);
  alloc_vec(&weight, size);
  rand_vec(x, size);
  rand_vec(weight, size);
  zero_vec(o, size);
  
  thablasStatus_t thablasStatus =  (o, x, weig  ht, size);
  thablasStatus = thaDNN_h2d_s_rmsnorm_v2(o, x, weight, size);
  // count time 
  std::chrono::time_point<std::chrono::high_resolution_clock> start_gpu, end_gpu;
  std::chrono::duration<double> estimate_gpu;

  start_gpu = std::chrono::high_resolution_clock::now();
  thablasStatus = thaDNN_h2d_s_rmsnorm(o, x, weight, size);
  end_gpu = std::chrono::high_resolution_clock::now();
  estimate_gpu = std::chrono::duration_cast<std::chrono::microseconds>(end_gpu - start_gpu);
  printf("GPU v1 time: %.10f\n", estimate_gpu.count() / 1.0);

  start_gpu = std::chrono::high_resolution_clock::now();
  thablasStatus = thaDNN_h2d_s_rmsnorm_v2(o, x, weight, size);
  end_gpu = std::chrono::high_resolution_clock::now();
  estimate_gpu = std::chrono::duration_cast<std::chrono::microseconds>(end_gpu - start_gpu);
  printf("GPU v2 time: %.10f\n", estimate_gpu.count() / 1.0);

  if (thablasStatus != THABLAS_STATUS_SUCCESS)
      return 0;

  float   *o_h;
  alloc_vec(&o_h, size);
  zero_vec(o_h, size);
  rmsnorm(o_h, x, weight, size);  

  bool is_valid = true;
  int cnt = 0, thr = 10;
  float eps = 1e-5;
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

  util_free((void*)x);
  util_free((void*)o);
  util_free((void*)weight);
  util_free((void *)o_h);

  if (is_valid) {
    printf("Validation: VALID\n"); fflush(stdout);
    return 1;
  } else {
    printf("Validation: INVALID\n"); fflush(stdout);
    return 0;
  }
}

bool test_thaDNN_h2d_s_softmax(int size)
{
  float *x_h, *x_d;
  alloc_vec(&x_h, size);
  alloc_vec(&x_d, size);
  rand_vec(x_h, size);
  memcpy(x_d, x_h, size * sizeof(float));
  

  // run on gpu 
  // auto start_gpu = std::chrono::high_resolution_clock::now();
  thablasStatus_t thablasStatus = thaDNN_h2d_s_softmax(x_d, size);
  // auto end_gpu = std::chrono::high_resolution_clock::now();
  // auto duration_gpu = std::chrono::duration_cast<std::chrono::microseconds>(end_gpu - start_gpu);


  if (thablasStatus != THABLAS_STATUS_SUCCESS)
      return 0;
  
  // auto start_cpu = std::chrono::high_resolution_clock::now();
  softmax(x_h, size);
  // auto end_cpu = std::chrono::high_resolution_clock::now();
  // auto duration_cpu = std::chrono::duration_cast<std::chrono::microseconds>(end_cpu - start_cpu);
  
  // print time with 10 decimal places
  // printf("GPU time: %.10f\n", duration_gpu.count() / 1000000.0);
  // printf("CPU time: %.10f\n", duration_cpu.count() / 1000000.0);

  bool is_valid = true;
  int cnt = 0, thr = 10;
  float eps = 1e-5;
  for (int i = 0; i < size; ++i) {
    float x_gpu = x_d[i];
    float x_ans = x_h[i];
    if (fabsf(x_gpu - x_ans) > eps &&
        (x_ans == 0 || fabsf((x_gpu - x_ans) / x_ans) > eps)) {
      ++cnt;
      if (cnt <= thr)
        printf("O[%d] : correct_value = %f, your_value = %f\n", i, x_ans, x_gpu);
      if (cnt == thr + 1)
        printf("Too many error, only first %d values are printed.\n", thr);
      is_valid = false;
    }
  }
  util_free((void*)x_d);
  util_free((void*)x_h);

  if (is_valid) {
    printf("Validation: VALID\n"); fflush(stdout);
    return 1;
  } else {
    printf("Validation: INVALID\n"); fflush(stdout);
    return 0;
  }
}


bool test_forward(int token, int pos)
{
  Transformer transformer;
  char checkpoint_path[64] = "/shared/erc/getpTA/main/modelbin/stories110M.bin";
  build_transformer(&transformer, checkpoint_path);
  Config* p = &transformer.config;
  
  int size = p->vocab_size;
  float* logits = forward(&transformer, token, pos);

  float* cpuLogits = (float*)malloc(size * sizeof(float));
  memcpy(cpuLogits, logits, size * sizeof(float));

  float*gpuLogits = nullptr;
  thablasStatus_t thablasStatus = thaDNN_h2d_s_forward(&transformer, token, pos, gpuLogits);
  if (thablasStatus != THABLAS_STATUS_SUCCESS)
    return 0;

  bool is_valid = true;
  int cnt = 0, thr = 10;
  float eps = 1e-4;
  for (int i = 0; i < p->vocab_size ; ++i) {
    float logit_gpu = gpuLogits[i];
    float logit_ans = cpuLogits[i];
    if (fabsf(logit_gpu - logit_ans) > eps && (logit_gpu == 0 || fabsf((logit_gpu - logit_ans) / logit_ans) > eps)) {
      ++cnt;
      if (cnt <= thr)
        printf("O[%d] : correct_value = %f, your_value = %f\n", i, logit_ans, logit_gpu);
      if (cnt == thr + 1)
        printf("Too many error, only first %d values are printed.\n", thr);
      is_valid = false;
    }
  }

  // TODO free transformer

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

  // test rmsnorm
  // all_valid = std::min(all_valid, test_thaDNN_h2d_s_rmsnorm(1));
  // assert(all_valid);
  // all_valid = std::min(all_valid, test_thaDNN_h2d_s_rmsnorm(111));
  // assert(all_valid);
  // all_valid = std::min(all_valid, test_thaDNN_h2d_s_rmsnorm(11111));
  // assert(all_valid);
  all_valid = std::min(all_valid, test_thaDNN_h2d_s_rmsnorm(1024*12));
  assert(all_valid);
  printf("RMSNORM PASSED\n");

  return 0;
}

