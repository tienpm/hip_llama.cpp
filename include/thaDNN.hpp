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
thablasStatus_t thaDNN_s_softmax(thablasHandle_t handle, float* output, float* x, int size);

// _h2d_ = host to device
// output, x allocated on Host
// only run on 1 devices
thablasStatus_t thaDNN_h2d_s_softmax(float* x, int size);

thablasStatus_t thaDNN_h2d_s_forward(Transformer* transformer, int token, int pos, float* &output_logits);
