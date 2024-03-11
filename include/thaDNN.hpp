#pragma once
#include "thaBLAS.hpp"
#include "utils.hpp"

// '_s_' = single persion (float)
// input: o, x, weight allocated on device
// input: size = 1 -> 16384
thablasStatus_t thaDNN_s_rmsnorm(thablasHandle_t handle, float* o, float* x, float* weight, int size);

// _h2d_ = host to device
// o, x, weight allocated on Host
// only run on 1 devices
thablasStatus_t thaDNN_h2d_s_rmsnorm(float* o, float* x, float* weight, int size);

// _s_ = single persion (float)
// input: output, x allocated on device
// input: size = 32000
thablasStatus_t thaDNN_s_softmax(thablasHandle_t handle, float* x, int size);

// _h2d_ = host to device
// x allocated on Host
// only run on 1 devices
thablasStatus_t thaDNN_h2d_s_softmax(float* x, int size);

thablasStatus_t thaDNN_h2d_s_forward(Transformer* transformer, int token, int pos, float* &output_logits);

thablasStatus_t thaDNN_h2d_s_rmsnorm_v2(float* o, float* x, float* weight, int size);

thablasStatus_t thaDNN_h2d_s_rmsnorm_v3(float* o, float* x, float* weight, int size);

// thablasStatus_t thaDNN_s_forward(thablasHandle_t handle, Transformer* transformer, int token, int pos, float* &output_logits);
thablasStatus_t thaDNN_s_forward(thablasHandle_t handle1, thablasHandle_t handle2, thablasHandle_t handle3, Transformer* transformer, int token, int pos, float* &output_logits);

thablasStatus_t thaDNN_h2d_s_rope(int dim, int head_size, int kv_dim, int pos, float *q, float *k);

thablasStatus_t thaDNN_h2d_s_swiglu(float *hb, float *hb2, int hidden_dim);
