#pragma once
#include "thaBLAS.hpp"
#include "hip_helper.hpp"

thablasStatus_t thaDNN_s_softmax_v2(thablasHandle_t handle, 
                                    float* x, 
                                    int size);
