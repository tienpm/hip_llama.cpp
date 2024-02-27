#pragma once
#include "thaBLAS.hpp"

// '_s_' = single persion (float)
// input: o, x, weight allocated on device
// input: size = 768 = 256 * 3
thablasStatus_t thaDNN_s_rmsnorm(thablasHandle_t handle, float* o, float* x, float* weight, int size);

// _h2d_ = host to device
// o, x, weight allocated on Host
// only run on 1 devices
thablasStatus_t thaDNN_h2d_s_rmsnorm(float* o, float* x, float* weight, int size);
