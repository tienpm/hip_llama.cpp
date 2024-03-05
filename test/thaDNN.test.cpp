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
  float eps = 1e-5;
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
  
  thablasStatus_t thablasStatus = thaDNN_h2d_s_rope(dim, head_size, kv_dim, pos, q, k);
  if (thablasStatus != THABLAS_STATUS_SUCCESS)
      return 0;
  RoPE_relative_positional_encoding(dim, head_size, kv_dim, pos, q_h, k_h);

  bool is_valid = true;
  int cnt = 0, thr = 50;
  float eps = 1e-4;
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


/*
*********************************************************************************************************
*  SwiGLU non-linearity
*********************************************************************************************************
*/

void SwiGLU( float *hb, float *hb2, int hidden_dim){
  for (int i = 0; i < hidden_dim; i++) {
    float val = hb[i];
    val *= (1.0f / (1.0f + expf(-val)));
    // elementwise multiply with w3(x)
    val *= hb2[i];
    hb[i] = val;
  }
}

bool test_swiglu(int hidden_dim){
  float *hb, *hb2;
  alloc_vec(&hb, hidden_dim);
  alloc_vec(&hb2, hidden_dim);
  rand_vec(hb, hidden_dim);
  rand_vec(hb2, hidden_dim);

  float *hb_h, *hb2_h;
  alloc_vec(&hb_h, hidden_dim);
  alloc_vec(&hb2_h, hidden_dim);
  memcpy(hb_h, hb, hidden_dim * sizeof(float));
  memcpy(hb2_h, hb2, hidden_dim * sizeof(float));

  SwiGLU(hb_h, hb2_h, hidden_dim);

  thablasStatus_t thablasStatus = thaDNN_h2d_s_swiglu(hb, hb2, hidden_dim);
  if (thablasStatus != THABLAS_STATUS_SUCCESS)
      return 0;

  bool is_valid = true;
  int cnt = 0, thr = 50;
  float eps = 1e-4;
  for (int i = 0; i < hidden_dim; ++i) {
    float hb_gpu = hb[i];
    float hb_ans = hb_h[i];
    if (fabsf(hb_gpu - hb_ans) > eps &&
        (hb_ans == 0 || fabsf((hb_gpu - hb_ans) / hb_ans) > eps)) {
      ++cnt;
      if (cnt <= thr)
        printf("HB[%d] : correct_value = %f, your_value = %f\n", i, hb_ans, hb_gpu);
      if (cnt == thr + 1)
        printf("Too many error, only first %d values are printed.\n", thr);
      is_valid = false;
    }
  }

  util_free((void*)hb);
  util_free((void*)hb2);
  util_free((void *)hb_h);
  util_free((void *)hb2_h);

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

  // test RoPE_relative_positional_encoding
  all_valid = std::min(all_valid, test_RoPE_relative_positional_encoding(256, 16, 64, 0));
  assert(all_valid);
  all_valid = std::min(all_valid, test_RoPE_relative_positional_encoding(2, 1, 2, 1));
  assert(all_valid);
  all_valid = std::min(all_valid, test_RoPE_relative_positional_encoding(16384, 256, 11111, 512));
  assert(all_valid);
  all_valid = std::min(all_valid, test_RoPE_relative_positional_encoding(2222, 333, 2111, 111));
  assert(all_valid);
  printf("RoPE_relative_positional_encoding PASSED\n");

  // test SwiGLU
  all_valid = std::min(all_valid, test_swiglu(256));
  assert(all_valid);
  all_valid = std::min(all_valid, test_swiglu(1));
  assert(all_valid);
  all_valid = std::min(all_valid, test_swiglu(1000000));
  assert(all_valid);
  all_valid = std::min(all_valid, test_swiglu(33333));
  assert(all_valid);
  printf("SwiGLU PASSED\n");

  return 0;
}
