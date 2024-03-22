#pragma once
#include "thaBLAS.hpp"
#include "utils.hpp"
#include "thaDNN/thaDNN_rmsnorm.hpp"
#include "thaDNN/thaDNN_rope.hpp"
#include "thaDNN/thaDNN_mha.hpp"
#include "thaDNN/thaDNN_softmax.hpp"
#include "thaDNN/thaDNN_swiglu.hpp"

#include <omp.h>

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
// thablasStatus_t thaDNN_s_rmsnorm(thablasHandle_t handle, float* o, float* x, float* weight, int size);

// _h2d_ = host to device
// o, x, weight allocated on Host
// only run on 1 devices
// thablasStatus_t thaDNN_h2d_s_rmsnorm(float* o, float* x, float* weight, int size);
//
// thablasStatus_t thaDNN_h2d_s_rmsnorm_v2(float* o, float* x, float* weight, int size);
//
// thablasStatus_t thaDNN_h2d_s_rmsnorm_v3(float* o, float* x, float* weight, int size);
/*
 * ============================== SOFTMAX ======================================
 * */


/*
 * =================== Rotational Position Embeding (RoPE) ========================
 * */

/*
 * ============================== SWIGLU ======================================
 * */


/*
 * ============================== FORWARD ======================================
 * */


// Forward
// thablasStatus_t thaDNN_s_forward_batch(thablasHandle_t handle1, thablasHandle_t handle2, thablasHandle_t handle3, int n_batches, Config *p, TransformerWeights* w, RunState* s, int token[], int pos[], float* output_logits[]);

thablasStatus_t thaDNN_s_forward_batch(thablasHandle_t handle1, thablasHandle_t handle2, thablasHandle_t handle3, int n_batches, Config *p, TransformerWeights* w, RunState* s_batch, int token[], int pos[], float* logits_host);

// thablasStatus_t thaDNN_s_forward_70B(thablasHandle_t handle, int batch_size, Config *p, TransformerWeights* h_w, RunState* h_s, TransformerWeights* d_w, RunState* s_batch, int token[], int pos[], float* logits_host);
thablasStatus_t thaDNN_s_forward_70B(thablasHandle_t handle, int batch_size, Config *p, TransformerWeights* h_w[], RunState* h_s, TransformerWeights* d_w, RunState* d_s, int token[], int pos[], float* logits_host);

thablasStatus_t thaDNN_s_forward_batch_pipe_line(thablasHandle_t handle[], int n_devices, int n_batches, Transformer* transformer_d[], int token[], int pos[], float* logits_host);

thablasStatus_t thaDNN_s_forward_batch_multiple_pipe_line(thablasHandle_t handle[], int host_thread_id, int n_host_threads, int n_devices, int batch_size, Config* p, TransformerWeights* w[], RunState* s_batch[], int token[], int pos[], float* logits_host, int* host_thread_status, int* device_host_thread, omp_lock_t *device_mtx);

thablasStatus_t thaDNN_s_forward_batch_multiple_pipe_line_layer_swap(thablasHandle_t handle[], int thread_id, int n_host_threads, int n_devices, int batch_size, int n_buffer_words, 
                                                                     Config* p, TransformerWeights* w[], RunState* s_batch[], RunState* s_host_batch[], int token[], int pos[], 
                                                                     float* logits_host, omp_lock_t *device_locks);                                                               