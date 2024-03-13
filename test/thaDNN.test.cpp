#include "hip_helper.hpp"
#include "thaBLAS.hpp"
#include "thaDNN.hpp"
#include "seq.hpp"
#include "utils.hpp"

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
  
  thablasStatus_t thablasStatus;
  // thablasStatus = thaDNN_h2d_s_rmsnorm(o, x, weight, size);
  thablasStatus = thaDNN_h2d_s_rmsnorm_v2(o, x, weight, size);
  // thablasStatus = thaDNN_h2d_s_rmsnorm_v3(o, x, weight, size);

  float   *o_h;
  alloc_vec(&o_h, size);
  zero_vec(o_h, size);
  rmsnorm(o_h, x, weight, size);  

  // count time 
  // std::chrono::time_point<std::chrono::high_resolution_clock> start_gpu, end_gpu;
  // std::chrono::duration<double> estimate_gpu;

  // start_gpu = std::chrono::high_resolution_clock::now();
  // thablasStatus = thaDNN_h2d_s_rmsnorm(o, x, weight, size);
  // end_gpu = std::chrono::high_resolution_clock::now();
  // estimate_gpu = std::chrono::duration_cast<std::chrono::microseconds>(end_gpu - start_gpu);
  // printf("GPU v1 time: %.10f\n", estimate_gpu.count() / 1.0);

  // start_gpu = std::chrono::high_resolution_clock::now();
  // thablasStatus = thaDNN_h2d_s_rmsnorm_v2(o, x, weight, size);
  // end_gpu = std::chrono::high_resolution_clock::now();
  // estimate_gpu = std::chrono::duration_cast<std::chrono::nanoseconds>(end_gpu - start_gpu);
  // printf("GPU rms_norm_v2 time: %.10f\n", estimate_gpu.count() / 1.0);

  // start_gpu = std::chrono::high_resolution_clock::now();
  // thablasStatus = thaDNN_h2d_s_rmsnorm_v3(o, x, weight, size);
  // end_gpu = std::chrono::high_resolution_clock::now();
  // estimate_gpu = std::chrono::duration_cast<std::chrono::microseconds>(end_gpu - start_gpu);
  // printf("GPU v3 time: %.10f\n", estimate_gpu.count() / 1.0);

  // start_gpu = std::chrono::high_resolution_clock::now();
  // rmsnorm(o_h, x, weight, size);
  // end_gpu = std::chrono::high_resolution_clock::now();
  // estimate_gpu = std::chrono::duration_cast<std::chrono::nanoseconds>(end_gpu - start_gpu);
  // printf("CPU rms_norm time: %.10f\n", estimate_gpu.count() / 1.0);

  if (thablasStatus != THABLAS_STATUS_SUCCESS)
      return 0;

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


bool test_h2d_forward(int token, int pos)
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

bool test_gpu_forward(int token, int pos)
{
  Transformer transformer;
  char checkpoint_path[64] = "/shared/erc/getpTA/main/modelbin/stories110M.bin";
  build_transformer(&transformer, checkpoint_path);
  Config* p = &transformer.config;

  thablasHandle_t handle1, handle2, handle3;
  thablasCreate(&handle1);
  thablasCreate(&handle2);
  thablasCreate(&handle3);
  Transformer *transformer_d = nullptr;
  copy_transformer_to_device(handle1, &transformer, transformer_d);
  
  int size = p->vocab_size;
  float* cpuLogits;
  float* gpuLogits;
  float* logits = forward(&transformer, token, pos);
  alloc_vec(&cpuLogits, size);
  alloc_vec(&gpuLogits, size);
  memcpy(cpuLogits, logits, size * sizeof(float));

  thablasStatus_t thablasStatus = thaDNN_s_forward(handle1, handle2, handle3, transformer_d, token, pos, logits);
  if (thablasStatus != THABLAS_STATUS_SUCCESS)
    return 0;
  CHECK_HIP(hipMemcpy(gpuLogits, logits, size * sizeof(float), hipMemcpyDeviceToHost));

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



/*
*********************************************************************************************************
*  RoPE relative positional encoding
*********************************************************************************************************
*/

void RoPE_relative_positional_encoding(int dim, int head_size, int kv_dim, int pos, float *q, float *k) 
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


bool test_thaDNN_h2d_s_softmax_v2(int size)
{
    // create a random vector
    float *input_cpu;
    float *input_gpu;
    
    alloc_vec(&input_cpu, size);
    alloc_vec(&input_gpu, size);
    rand_vec(input_cpu, size);

    // copy the vector to the gpu
    memcpy(input_gpu, input_cpu, size * sizeof(float));

    // run the function on the gpu
    
    //init start, end, duration for time count
    std :: chrono :: time_point < std :: chrono :: high_resolution_clock > start_gpu, end_gpu;
    std :: chrono :: duration < double > estimate_gpu;
    thablasStatus_t thablasStatus;  

    // warm up kernel
    thablasStatus = thaDNN_h2d_s_softmax_v2(input_gpu, size);
    thablasStatus = thaDNN_h2d_s_softmax(input_gpu, size);

    // evaluate time
    start_gpu = std :: chrono :: high_resolution_clock :: now();
    thablasStatus = thaDNN_h2d_s_softmax(input_gpu, size);
    end_gpu = std :: chrono :: high_resolution_clock :: now();
    estimate_gpu = std :: chrono :: duration_cast < std :: chrono :: microseconds > (end_gpu - start_gpu);
    printf("GPU v1 time: %.10f\n", estimate_gpu.count() / 1.0);

    memcpy(input_gpu, input_cpu, size * sizeof(float));
    start_gpu = std :: chrono :: high_resolution_clock :: now();
    thablasStatus = thaDNN_h2d_s_softmax_v2(input_gpu, size);
    end_gpu = std :: chrono :: high_resolution_clock :: now();
    estimate_gpu = std :: chrono :: duration_cast < std :: chrono :: microseconds > (end_gpu - start_gpu);
    printf("GPU v2 time: %.10f\n", estimate_gpu.count() / 1.0);

    if (thablasStatus != THABLAS_STATUS_SUCCESS) {
      return false;
    }

    // run the function on the cpu
    softmax(input_cpu, size);

    // compare the results
    bool is_valid = true;
    int cnt = 0, thr = 70;
    float eps = 1e-3;

    for (int i = 0; i < size; i++) 
    {
      float o_gpu = input_gpu[i];
      float o_ans = input_cpu[i];
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

  // test rmsnorm
  all_valid = std::min(all_valid, test_thaDNN_h2d_s_rmsnorm(1));
  assert(all_valid);
  all_valid = std::min(all_valid, test_thaDNN_h2d_s_rmsnorm(111));
  assert(all_valid);
  all_valid = std::min(all_valid, test_thaDNN_h2d_s_rmsnorm(11111));
  assert(all_valid);
  all_valid = std::min(all_valid, test_thaDNN_h2d_s_rmsnorm(256*256));
  printf("RMS Norm PASSED\n");

  // test softmax v2
  all_valid = std::min(all_valid, test_thaDNN_h2d_s_softmax_v2(1));
  assert(all_valid);
  all_valid = std::min(all_valid, test_thaDNN_h2d_s_softmax_v2(111));
  assert(all_valid);
  all_valid = std::min(all_valid, test_thaDNN_h2d_s_softmax_v2(11111));
  assert(all_valid);
  all_valid = std::min(all_valid, test_thaDNN_h2d_s_softmax_v2(32000));
  printf("Softmax V2 PASSED\n");

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

  // test H2D forward
  all_valid = test_h2d_forward(0, 0);
  assert(all_valid);
  all_valid = test_h2d_forward(3, 4);
  assert(all_valid);
  all_valid = test_h2d_forward(4, 4);
  assert(all_valid);
  printf("FORWARD H2D PASSED\n");

  // test H2D forward
  all_valid = test_gpu_forward(0, 0);
  assert(all_valid);
  all_valid = test_gpu_forward(3, 4);
  assert(all_valid);
  all_valid = test_gpu_forward(4, 4);
  assert(all_valid);
  all_valid = test_gpu_forward(64, 64);
  assert(all_valid);
  printf("FORWARD GPU PASSED\n");

  return 0;
}

