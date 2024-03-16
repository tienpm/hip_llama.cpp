#pragma once
#include "thaBLAS.hpp"
#include "utils.hpp"

/*! @enum thaDNNStatus_t
 * Error codes that are returned by all MIOpen API calls.
 */
typedef enum
{
    thaDNNStatusSuccess              = 0, /*!< No errors */
    thaDNNStatusNotInitialized       = 1, /*!< Data not initialized. */
    thaDNNStatusInvalidValue         = 2, /*!< Incorrect variable value. */
    thaDNNStatusBadParm              = 3, /*!< Incorrect parameter detected. */
    thaDNNStatusAllocFailed          = 4, /*!< Memory allocation error. */
    thaDNNStatusInternalError        = 5, /*!< MIOpen failure. */
    thaDNNStatusNotImplemented       = 6, /*!< Use of unimplemented feature. */
    thaDNNStatusUnknownError         = 7, /*!< Unknown error occurred. */
    thaDNNStatusUnsupportedOp        = 8, /*!< Unsupported operator for fusion. */
    thaDNNStatusGpuOperationsSkipped = 9, /*!< This is not an error. */
    thaDNNStatusVersionMismatch = 10, /*!< Version mismatch of the supplied binary data argment. */
} thaDNNStatus_t;

/*
 * ============================== RMSNORM ======================================
 * */

// '_s_' = single persion (float)
// input: o, x, weight allocated on device
// input: size = 1 -> 16384
thablasStatus_t thaDNN_s_rmsnorm(thablasHandle_t handle, float* o, float* x, float* weight, int size);

// _h2d_ = host to device
// o, x, weight allocated on Host
// only run on 1 devices
thablasStatus_t thaDNN_h2d_s_rmsnorm(float* o, float* x, float* weight, int size);

thablasStatus_t thaDNN_h2d_s_rmsnorm_v2(float* o, float* x, float* weight, int size);

thablasStatus_t thaDNN_h2d_s_rmsnorm_v3(float* o, float* x, float* weight, int size);
/*
 * ============================== SOFTMAX ======================================
 * */

// _s_ = single persion (float)
// input: output, x allocated on device
// input: size = 32000
thablasStatus_t thaDNN_s_softmax(thablasHandle_t handle, float* x, int size);

// _h2d_ = host to device
// x allocated on Host
// only run on 1 devices
thablasStatus_t thaDNN_h2d_s_softmax(float* x, int size);

/*
 * =================== Rotational Position Embeding (RoPE) ========================
 * */
thablasStatus_t thaDNN_h2d_s_rope(int dim, int head_size, int kv_dim, int pos, float *q, float *k);

/*
 * ============================== SWIGLU ======================================
 * */


thablasStatus_t thaDNN_h2d_s_swiglu(float *hb, float *hb2, int hidden_dim);

/*
 * ============================== FORWARD ======================================
 * */
thablasStatus_t thaDNN_h2d_s_forward(Transformer* transformer, int token, int pos, float* &output_logits);

// thablasStatus_t thaDNN_s_forward(thablasHandle_t handle, Transformer* transformer, int token, int pos, float* &output_logits);
thablasStatus_t thaDNN_s_forward(thablasHandle_t handle1, thablasHandle_t handle2, thablasHandle_t handle3, Transformer* transformer, int token, int pos, float* &output_logits);

thablasStatus_t thaDNN_h2d_s_softmax_v2(float* x, int size);

// thablasStatus_t thaDNN_s_forward_batch(thablasHandle_t handle1, thablasHandle_t handle2, thablasHandle_t handle3, int n_batches, Config *p, TransformerWeights* w, RunState* s, int token[], int pos[], float* output_logits[]);
thablasStatus_t thaDNN_s_forward_batch(thablasHandle_t handle1, thablasHandle_t handle2, thablasHandle_t handle3, int n_batches, Config *p, TransformerWeights* w, RunState* s_batch, int token[], int pos[], float* logits_host);

thablasStatus_t thaDNN_s_multiheads_1_v2_batch(thablasHandle_t handle, int n_batches, int pos[], int pos_d[], int n_heads, int n_layers, float* s_q_batch, float* s_att_batch, float* s_key_cache_batch, int head_size, int seq_len, int loff, int kv_dim, int dim, int kv_mul);

thablasStatus_t thaDNN_s_rmsnorm_v2_batch(thablasHandle_t handle, int n_batches, float* o_batch, float* x_batch, float* weight, int size, int dim);

thablasStatus_t thaDNN_s_multiheads_2_batch(thablasHandle_t handle, int n_batches, float* s_att_batch, int size_batch[], int seq_len, int n_heads);

thablasStatus_t thaDNN_s_matmulvec_v2_batch(thablasHandle_t handle, int n_batches, float *C_batch, float *B_batch, float *A, int K, int M, int Coff, int has_pos, int pos_d[], int C_batch_size, int B_batch_size);

thablasStatus_t thaDNN_s_multiheads_3_v2_batch(thablasHandle_t handle, int n_batches, int pos_d[], int n_heads, float *s_xb_batch, float *s_att_batch, float *s_value_cache_batch, int head_size, int seq_len, int loff, int kv_dim, int kv_mul, int dim, int n_layers);
