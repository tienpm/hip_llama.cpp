#pragma once
#include "thaBLAS.hpp"


thablasStatus_t thaDNN_s_multiheads_1_v1_batch(thablasHandle_t handle, int n_batches, int pos[], int pos_d[], int n_heads, int n_layers, float* s_q_batch, float* s_att_batch, float* s_key_cache_batch, int head_size, int seq_len, int loff, int kv_dim, int dim, int kv_mul);
thablasStatus_t thaDNN_s_multiheads_2_v1_batch(thablasHandle_t handle, int n_batches, float* s_att_batch, int size_batch[], int seq_len, int n_heads);
thablasStatus_t thaDNN_s_multiheads_3_v1_batch(thablasHandle_t handle, int n_batches, int pos_d[], int n_heads, float *s_xb_batch, float *s_att_batch, float *s_value_cache_batch, int head_size, int seq_len, int loff, int kv_dim, int kv_mul, int dim, int n_layers);

thablasStatus_t thaDNN_s_multiheads_1_v2_batch(thablasHandle_t handle,
                                               int batch_size,
                                               int pipe_size, 
                                               int pos[], 
                                               int pos_d[], 
                                               int n_heads, 
                                               float* s_q_batch, 
                                               float* s_att_batch, 
                                               float* s_key_cache_batch, 
                                               int head_size, 
                                               int n_words, 
                                               int kv_dim, 
                                               int dim, 
                                               int kv_mul);


// _s_ = single persion (float)
// input: output, x allocated on device
// input: size = 32000
thablasStatus_t thaDNN_s_multiheads_2_batch(thablasHandle_t handle, 
                                            int n_batches, 
                                            float* s_att_batch, 
                                            int size_batch[], 
                                            int seq_len, 
                                            int n_heads);



thablasStatus_t thaDNN_s_multiheads_3_v2_batch(thablasHandle_t handle, 
                                               int batch_size, int pos_d[], 
                                               int n_heads, 
                                               float *s_xb_batch, 
                                               float *s_att_batch, 
                                               float *s_value_cache_batch, 
                                               int head_size, int n_words, 
                                               int kv_dim, 
                                               int kv_mul, 
                                               int dim, 
                                               int pipe_size);
