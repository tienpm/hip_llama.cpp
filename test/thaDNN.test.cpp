#include "hip_helper.hpp"
#include "thaBLAS.hpp"
#include "thaDNN.hpp"
#include "seq.hpp"

#include <alloca.h>
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

  thablasStatus_t thablasStatus = thaDNN_h2d_s_rmsnorm(o, x, weight, size);
  if (thablasStatus != THABLAS_STATUS_SUCCESS)
      return 0;

  float *o_h;
  alloc_vec(&o_h, size);
  zero_vec(o_h, size);
  rmsnorm(o_h, x, weight, size);  

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
  float *o, *x;
  alloc_vec(&o, size);
  alloc_vec(&x, size);
  rand_vec(x, size);
  zero_vec(o, size);

  // run on gpu 
  // auto start_gpu = std::chrono::high_resolution_clock::now();
  thablasStatus_t thablasStatus = thaDNN_h2d_s_softmax(o, x, size);
  // auto end_gpu = std::chrono::high_resolution_clock::now();
  // auto duration_gpu = std::chrono::duration_cast<std::chrono::microseconds>(end_gpu - start_gpu);


  if (thablasStatus != THABLAS_STATUS_SUCCESS)
      return 0;

  float *o_h;
  alloc_vec(&o_h, size);
  zero_vec(o_h, size);
  memcpy(o_h, x, size * sizeof(float));
  
  // auto start_cpu = std::chrono::high_resolution_clock::now();
  softmax(o_h, size);
  // auto end_cpu = std::chrono::high_resolution_clock::now();
  // auto duration_cpu = std::chrono::duration_cast<std::chrono::microseconds>(end_cpu - start_cpu);
  
  // print time with 10 decimal places
  // printf("GPU time: %.10f\n", duration_gpu.count() / 1000000.0);
  // printf("CPU time: %.10f\n", duration_cpu.count() / 1000000.0);

  bool is_valid = true;
  int cnt = 0, thr = 10;
  float eps = 1e-4;
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
  util_free((void *)o_h);

  if (is_valid) {
    printf("Validation: VALID\n"); fflush(stdout);
    return 1;
  } else {
    printf("Validation: INVALID\n"); fflush(stdout);
    return 0;
  }
}


bool test_forward()
{
  Transformer transformer;
  char checkpoint_path[64] = "/shared/erc/getpTA/main/modelbin/stories110M.bin";
  build_transformer(&transformer, checkpoint_path);
  
  float* cpuLogits = forward(&transformer, 0, 0);
  float* gpuLogits = forward(&transformer, 0, 0);

  Config* p = &transformer.config;

  bool is_valid = true;
  int cnt = 0, thr = 10;
  float eps = 1e-4;
  for (int i = 0; i < p->vocab_size ; ++i) {
    float logit_gpu = gpuLogits[i];
    float logit_ans = cpuLogits[i];
    if (fabsf(logit_gpu - logit_ans) > eps &&
        (logit_gpu == 0 || fabsf((logit_gpu - logit_ans) / logit_ans) > eps)) {
      ++cnt;
      if (cnt <= thr)
        printf("O[%d] : correct_value = %f, your_value = %f\n", i, logit_ans, logit_gpu);
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


/*
*********************************************************************************************************
*  RoPE relative positional encoding
*********************************************************************************************************
*/

void RoPE_relative_positional_encoding(int dim,  int head_size, int kv_dim, int pos,  float *q, float *k) 
{
    // RoPE relative positional encoding: complex-valued rotate q and k in each head
    // int pos = pos;
    for (int i = 0; i < dim; i+=2) {
      int head_dim = i % head_size;
      float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
      float val = pos * freq;
      float fcr = cosf(val);
      float fci = sinf(val);
      int rotn = i < kv_dim ? 2 : 1; // how many vectors? 2 = q & k, 1 = q only
      for (int v = 0; v < rotn; v++) {
        float* vec = v == 0 ? q : k; // the vector to rotate (query or key)
        float v0 = vec[i];
        float v1 = vec[i+1];
        vec[i]   = v0 * fcr - v1 * fci;
        vec[i+1] = v0 * fci + v1 * fcr;
      }
    }
}

bool test_RoPE_relative_positional_encoding(int dim, int head_size, int kv_dim, int pos)
{
  float *q, *k;
  alloc_vec(&q, dim);
  alloc_vec(&k, dim);
  rand_vec(q, dim);
  rand_vec(k, dim);

  float *q_h, *k_h;
  alloc_vec(&q_h, dim);
  alloc_vec(&k_h, dim);
  memcpy(q_h, q, dim * sizeof(float));
  memcpy(k_h, k, dim * sizeof(float));

  RoPE_relative_positional_encoding(dim, head_size, kv_dim, pos, q_h, k_h);

  bool is_valid = true;
  int cnt = 0, thr = 50;
  float eps = 0.1;
  for (int i = 0; i < dim; ++i) {
    float q_gpu = q[i];
    float q_ans = q_h[i];
    if (fabsf(q_gpu - q_ans) > eps &&
        (q_ans == 0 || fabsf((q_gpu - q_ans) / q_ans) > eps)) {
      ++cnt;
      if (cnt <= thr)
        printf("Q[%d] : correct_value = %f, your_value = %f\n", i, q_ans, q_gpu);
      if (cnt == thr + 1)
        printf("Too many error, only first %d values are printed.\n", thr);
      is_valid = false;
    }
  }
  for (int i = 0; i < dim; ++i) {
    float k_gpu = k[i];
    float k_ans = k_h[i];
    if (fabsf(k_gpu - k_ans) > eps &&
        (k_ans == 0 || fabsf((k_gpu - k_ans) / k_ans) > eps)) {
      ++cnt;
      if (cnt <= thr)
        printf("K[%d] : correct_value = %f, your_value = %f\n", i, k_ans, k_gpu);
      if (cnt == thr + 1)
        printf("Too many error, only first %d values are printed.\n", thr);
      is_valid = false;
    }
  }

  util_free((void*)q);
  util_free((void*)k);
  util_free((void *)q_h);
  util_free((void *)k_h);

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

  // // test rmsnorm
  // all_valid = std::min(all_valid, test_thaDNN_h2d_s_rmsnorm(512));
  // assert(all_valid);
  // all_valid = std::min(all_valid, test_thaDNN_h2d_s_rmsnorm(768));
  // assert(all_valid);
  // all_valid = std::min(all_valid, test_thaDNN_h2d_s_rmsnorm(4096));
  // assert(all_valid);
  // all_valid = std::min(all_valid, test_thaDNN_h2d_s_rmsnorm(5120));
  // assert(all_valid);
  // all_valid = std::min(all_valid, test_thaDNN_h2d_s_rmsnorm(8192));
  // assert(all_valid);
  // all_valid = std::min(all_valid, test_thaDNN_h2d_s_rmsnorm(1));
  // assert(all_valid);
  // all_valid = std::min(all_valid, test_thaDNN_h2d_s_rmsnorm(111));
  // assert(all_valid);
  // all_valid = std::min(all_valid, test_thaDNN_h2d_s_rmsnorm(11111));
  // assert(all_valid);
  // all_valid = std::min(all_valid, test_thaDNN_h2d_s_rmsnorm(16384));
  // assert(all_valid);
  // printf("RMSNORM PASSED\n");

  // // test softmax
  // all_valid = std::min(all_valid, test_thaDNN_h2d_s_softmax(1));
  // assert(all_valid);
  // all_valid = std::min(all_valid, test_thaDNN_h2d_s_softmax(111));
  // assert(all_valid);
  // all_valid = std::min(all_valid, test_thaDNN_h2d_s_softmax(11111));
  // assert(all_valid);
  // all_valid = std::min(all_valid, test_thaDNN_h2d_s_softmax(32000));
  // assert(all_valid);
  // printf("SOFTMAX PASSED\n");

  // // test forward
  // all_valid = test_forward();
  // assert(all_valid);

  // test RoPE_relative_positional_encoding
  all_valid = std::min(all_valid, test_RoPE_relative_positional_encoding(768, 64, 768, 0));
  assert(all_valid);
  all_valid = std::min(all_valid, test_RoPE_relative_positional_encoding(768, 64, 768, 2));
  assert(all_valid);
  all_valid = std::min(all_valid, test_RoPE_relative_positional_encoding(768, 64, 768, 1));
  assert(all_valid);
  all_valid = std::min(all_valid, test_RoPE_relative_positional_encoding(5120, 128, 256, 1));
  assert(all_valid);
  all_valid = std::min(all_valid, test_RoPE_relative_positional_encoding(8192, 128, 256, 5));
  assert(all_valid);
  all_valid = std::min(all_valid, test_RoPE_relative_positional_encoding(1, 1, 1, 1));
  printf("RoPE_relative_positional_encoding PASSED\n");



  return 0;
}
