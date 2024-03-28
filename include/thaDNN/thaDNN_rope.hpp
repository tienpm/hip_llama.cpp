#pragma once
#include "thaBLAS.hpp"


thablasStatus_t thaDNN_s_rope(thablasHandle_t* handle, 
                              int dim, 
                              int head_size, 
                              int kv_dim, 
                              int pos, 
                              float *q, 
                              float *k);
