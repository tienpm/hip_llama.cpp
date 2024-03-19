#pragma once
#include "thaBLAS.hpp"

thablasStatus_t thaDNN_s_rmsnorm_v2_batch(thablasHandle_t handle, 
                                          int n_batches, 
                                          float* o_batch, 
                                          float* x_batch, 
                                          float* weight, 
                                          int size, 
                                          int dim); 
