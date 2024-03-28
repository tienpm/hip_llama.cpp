#pragma once
#include "thaBLAS.hpp"

thablasStatus_t thaDNN_s_swiglu(thablasHandle_t* handle, 
                                float *hb, 
                                float *hb2, 
                                int hidden_dim);
