#include "thaDNN.hpp"
#include "thaBLAS.hpp"
#include "hip_helper.hpp"
#include "seq.hpp"

#include <hip/hip_runtime.h>
#include <omp.h>
#include <mutex>


/*
*********************************************************************************************************
* rmsnorm
*********************************************************************************************************
*/

// __device__ float maxReduce_device(volatile float* data, int n) {
//   int lane_x = threadIdx.x;
//   __shared__ float max_value;

//   float val = -3.402e+38;

//   for (int i = lane_x; i < n; i += blockDim.x) {
//     val = max(val, data[i]);
//   }

//   val = block_reduce_max(val);

//   if (lane_x == 0) max_value = val;

//   __syncthreads();

//   // if (blockIdx.x == 0 && threadIdx.x == 0) max_value_return[0] = max_value;
//   return max_value;
// }

// /* 
//  * rmsnorm using shuffle and reduce 
//  * */
//


/*
*********************************************************************************************************
* softmax
*********************************************************************************************************
*/




/* 
 * softmax_v2: reduction using shuffle and reduce 
 * */


/*
*********************************************************************************************************
*  RoPE relative positional encoding
*********************************************************************************************************
*/





/*
*********************************************************************************************************
*  SwiGLU non-linearity
*********************************************************************************************************
*/



/*
*********************************************************************************************************
* FORWARD
*********************************************************************************************************
*/

thablasStatus_t thaDNN_s_forward_batch(thablasHandle_t handle1, thablasHandle_t handle2, thablasHandle_t handle3, int batch_size, Config *p, TransformerWeights* w, RunState* s_batch, int token[], int pos[], float* logits_host) {
    float *x[batch_size];
    int dim = p->dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads; 
    int kv_mul = p->n_heads / p->n_kv_heads; // integer multiplier of the kv sharing in multiquery
    int hidden_dim =  p->hidden_dim;
    int head_size = dim / p->n_heads;
    for(int b=0 ; b<batch_size ; ++b)
        x[b] = s_batch->x + b * dim;

    int *pos_d;
    CHECK_HIP(hipMalloc(&pos_d, batch_size * sizeof(int)));
    CHECK_HIP(hipMemcpy(pos_d, pos, batch_size * sizeof(int), hipMemcpyHostToDevice));

    thablasStatus_t thablas_status = THABLAS_STATUS_SUCCESS;

    // copy the token embedding into x
    float* content_row[batch_size];
    for(int b=0 ; b<batch_size ; ++b)
    {
        content_row[b] = w->token_embedding_table + token[b] * dim;
        // memcpy(x[b], content_row[b], dim*sizeof(float)); // TODO: copy device to device
        CHECK_HIP(hipMemcpy(s_batch->x + b * dim, content_row[b], dim * sizeof(float), hipMemcpyDeviceToDevice));
    }

    // forward all the layers
    for(unsigned long long l = 0; l < p->n_layers; l++) {
        thablas_status = thaDNN_s_rmsnorm_v2_batch(handle1, batch_size, s_batch->xb, s_batch->x, w->rms_att_weight + l*dim, dim, dim);

        int loff = l * p->seq_len * kv_dim;

        thablas_status = thaBLAS_s_matmul_batch(handle1, batch_size, s_batch->q, s_batch->xb, w->wq + l*dim*dim, dim, dim, 0, 0, pos_d, dim, dim);
        thablas_status = thaBLAS_s_matmul_batch(handle1, batch_size, s_batch->key_cache, s_batch->xb, w->wk + l*dim*kv_dim, dim, kv_dim, loff, kv_dim, pos_d, p->n_layers * p->seq_len * kv_dim, dim);
        thablas_status = thaBLAS_s_matmul_batch(handle1, batch_size, s_batch->value_cache, s_batch->xb, w->wv + l*dim*kv_dim, dim, kv_dim, loff, kv_dim, pos_d, p->n_layers * p->seq_len * kv_dim, dim);

        for(int b=0 ; b<batch_size ; ++b)
            thablas_status = thaDNN_s_rope(handle1, dim, head_size, kv_dim, pos[b], s_batch->q + b * dim, s_batch->key_cache + loff + pos[b] * kv_dim + b * p->n_layers * p->seq_len * kv_dim);

        // multi-head attention
        thablas_status = thaDNN_s_multiheads_1_v2_batch(handle1, batch_size, pos, pos_d, p->n_heads, p->n_layers, s_batch->q, s_batch->att, s_batch->key_cache, head_size, p->seq_len, loff, kv_dim, dim, kv_mul);
        thablas_status = thaDNN_s_multiheads_2_batch(handle1, batch_size, s_batch->att, pos_d, p->seq_len, p->n_heads);
        thablas_status = thaDNN_s_multiheads_3_v2_batch(handle1, batch_size, pos_d, p->n_heads, s_batch->xb, s_batch->att, s_batch->value_cache, head_size, p->seq_len, loff, kv_dim, kv_mul, dim, p->n_layers);

        thablas_status = thaBLAS_s_matmul_batch(handle1, batch_size, s_batch->xb2, s_batch->xb, w->wo + l*dim*dim, dim, dim, 0, 0, pos_d, dim, dim);

        for(int b=0 ; b<batch_size ; ++b)
            thablas_status = thaBLAS_s_vecaddvec(handle1, x[b], s_batch->xb2 + b * dim, dim);

        thablas_status = thaDNN_s_rmsnorm_v2_batch(handle1, batch_size, s_batch->xb, s_batch->x, w->rms_ffn_weight + l*dim, dim, dim);

        thablas_status = thaBLAS_s_matmul_batch(handle1, batch_size, s_batch->hb, s_batch->xb, w->w1 + l*dim*hidden_dim, dim, hidden_dim, 0, 0, pos_d, hidden_dim, dim);
        thablas_status = thaBLAS_s_matmul_batch(handle1, batch_size, s_batch->hb2, s_batch->xb, w->w3 + l*dim*hidden_dim, dim, hidden_dim, 0, 0, pos_d, hidden_dim, dim);

        for(int b=0 ; b<batch_size ; ++b)
            thablas_status = thaDNN_s_swiglu(handle1, s_batch->hb + b * hidden_dim, s_batch->hb2 + b * hidden_dim, hidden_dim);

        thablas_status = thaBLAS_s_matmul_batch(handle1, batch_size, s_batch->xb, s_batch->hb, w->w2 + l*dim*hidden_dim, hidden_dim, dim, 0, 0, pos_d, dim, hidden_dim);

        for(int b=0 ; b<batch_size ; ++b)
            thablas_status = thaBLAS_s_vecaddvec(handle1, x[b], s_batch->xb + b * dim, dim);
    }

    thablas_status = thaDNN_s_rmsnorm_v2_batch(handle1, batch_size, s_batch->x, s_batch->x, w->rms_final_weight, dim, dim);
    thablas_status = thaBLAS_s_matmul_batch(handle1, batch_size, logits_host, s_batch->x, w->wcls, dim, p->vocab_size, 0, 0, pos_d, p->vocab_size, dim);
    // for(int b=0 ; b<batch_size ; ++b)    
    //     output_logits[b] = s_batch->logits + b * p->vocab_size;
        
    CHECK_HIP(hipDeviceSynchronize());
    CHECK_HIP(hipFree(pos_d));
    return thablas_status;
}

thablasStatus_t thaDNN_s_forward_batch_pipe_line(thablasHandle_t handle[], int n_devices, int batch_size, Transformer* transformer_d[], int token[], int pos[], float* logits_host) {
    TransformerWeights* w[n_devices];
    RunState* s_batch[n_devices];
    for(int gid=0 ; gid<n_devices ; ++gid)
    {
        w[gid] = &transformer_d[gid]->weights;
        s_batch[gid] = &transformer_d[gid]->state;
    }
    Config* p = &transformer_d[0]->config;
    int dim = p->dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads; 
    int kv_mul = p->n_heads / p->n_kv_heads; // integer multiplier of the kv sharing in multiquery
    int hidden_dim =  p->hidden_dim;
    int head_size = dim / p->n_heads;
    int pipe_size = p->n_layers / n_devices;

    thablasStatus_t thablas_status = THABLAS_STATUS_SUCCESS;

    // copy the token embedding into x
    CHECK_HIP(hipSetDevice(0));
    float* content_row[batch_size];
    for(int b=0 ; b<batch_size ; ++b)
    {
        content_row[b] = w[0]->token_embedding_table + token[b] * dim;
        // memcpy(x[b], content_row[b], dim*sizeof(float)); // TODO: copy device to device
        CHECK_HIP(hipMemcpy(s_batch[0]->x + b * dim, content_row[b], dim * sizeof(float), hipMemcpyDeviceToDevice));
    }

    for(int gid=0; gid<n_devices ; ++gid) { // loop each device 
        CHECK_HIP(hipSetDevice(gid));
        int *pos_d;
        CHECK_HIP(hipMalloc(&pos_d, batch_size * sizeof(int)));
        CHECK_HIP(hipMemcpy(pos_d, pos, batch_size * sizeof(int), hipMemcpyHostToDevice));
        // forward all the layers
        for(unsigned long long l = 0; l < pipe_size; l++) {
            thablas_status = thaDNN_s_rmsnorm_v2_batch(handle[gid], batch_size, s_batch[gid]->xb, s_batch[gid]->x, w[gid]->rms_att_weight + l*dim, dim, dim);

            int loff = l * p->seq_len * kv_dim;

            thablas_status = thaBLAS_s_matmul_batch(handle[gid], batch_size, s_batch[gid]->q, s_batch[gid]->xb, w[gid]->wq + l*dim*dim, dim, dim, 0, 0, pos_d, dim, dim);
            thablas_status = thaBLAS_s_matmul_batch(handle[gid], batch_size, s_batch[gid]->key_cache, s_batch[gid]->xb, w[gid]->wk + l*dim*kv_dim, dim, kv_dim, loff, kv_dim, pos_d, pipe_size * p->seq_len * kv_dim, dim);
            thablas_status = thaBLAS_s_matmul_batch(handle[gid], batch_size, s_batch[gid]->value_cache, s_batch[gid]->xb, w[gid]->wv + l*dim*kv_dim, dim, kv_dim, loff, kv_dim, pos_d, pipe_size * p->seq_len * kv_dim, dim);

            for(int b=0 ; b<batch_size ; ++b)
                thablas_status = thaDNN_s_rope(handle[gid], dim, head_size, kv_dim, pos[b], s_batch[gid]->q + b * dim, s_batch[gid]->key_cache + loff + pos[b] * kv_dim + b * pipe_size * p->seq_len * kv_dim);

            // multi-head attention
            thablas_status = thaDNN_s_multiheads_1_v2_batch(handle[gid], batch_size, pos, pos_d, p->n_heads, pipe_size, s_batch[gid]->q, s_batch[gid]->att, s_batch[gid]->key_cache, head_size, p->seq_len, loff, kv_dim, dim, kv_mul);
            thablas_status = thaDNN_s_multiheads_2_batch(handle[gid], batch_size, s_batch[gid]->att, pos_d, p->seq_len, p->n_heads);
            thablas_status = thaDNN_s_multiheads_3_v2_batch(handle[gid], batch_size, pos_d, p->n_heads, s_batch[gid]->xb, s_batch[gid]->att, s_batch[gid]->value_cache, head_size, p->seq_len, loff, kv_dim, kv_mul, dim, pipe_size);

            thablas_status = thaBLAS_s_matmul_batch(handle[gid], batch_size, s_batch[gid]->xb2, s_batch[gid]->xb, w[gid]->wo + l*dim*dim, dim, dim, 0, 0, pos_d, dim, dim);

            for(int b=0 ; b<batch_size ; ++b)
                thablas_status = thaBLAS_s_vecaddvec(handle[gid], s_batch[gid]->x + b * dim, s_batch[gid]->xb2 + b * dim, dim);

            thablas_status = thaDNN_s_rmsnorm_v2_batch(handle[gid], batch_size, s_batch[gid]->xb, s_batch[gid]->x, w[gid]->rms_ffn_weight + l*dim, dim, dim);

            thablas_status = thaBLAS_s_matmul_batch(handle[gid], batch_size, s_batch[gid]->hb, s_batch[gid]->xb, w[gid]->w1 + l*dim*hidden_dim, dim, hidden_dim, 0, 0, pos_d, hidden_dim, dim);
            thablas_status = thaBLAS_s_matmul_batch(handle[gid], batch_size, s_batch[gid]->hb2, s_batch[gid]->xb, w[gid]->w3 + l*dim*hidden_dim, dim, hidden_dim, 0, 0, pos_d, hidden_dim, dim);

            for(int b=0 ; b<batch_size ; ++b)
                thablas_status = thaDNN_s_swiglu(handle[gid], s_batch[gid]->hb + b * hidden_dim, s_batch[gid]->hb2 + b * hidden_dim, hidden_dim);

            thablas_status = thaBLAS_s_matmul_batch(handle[gid], batch_size, s_batch[gid]->xb, s_batch[gid]->hb, w[gid]->w2 + l*dim*hidden_dim, hidden_dim, dim, 0, 0, pos_d, dim, hidden_dim);

            for(int b=0 ; b<batch_size ; ++b)
                thablas_status = thaBLAS_s_vecaddvec(handle[gid], s_batch[gid]->x + b * dim, s_batch[gid]->xb + b * dim, dim);
        }

        int next_device = (gid + 1) % n_devices;
        CHECK_HIP(hipMemcpy(s_batch[next_device]->x, s_batch[gid]->x, batch_size * dim * sizeof(float), hipMemcpyDeviceToDevice));
        CHECK_HIP(hipDeviceSynchronize());
        CHECK_HIP(hipFree(pos_d));
    }

    CHECK_HIP(hipSetDevice(0));
    int *pos_d;
    CHECK_HIP(hipMalloc(&pos_d, batch_size * sizeof(int)));
    CHECK_HIP(hipMemcpy(pos_d, pos, batch_size * sizeof(int), hipMemcpyHostToDevice));

    thablas_status = thaDNN_s_rmsnorm_v2_batch(handle[0], batch_size, s_batch[0]->x, s_batch[0]->x, w[0]->rms_final_weight, dim, dim);
    thablas_status = thaBLAS_s_matmul_batch(handle[0], batch_size, logits_host, s_batch[0]->x, w[0]->wcls, dim, p->vocab_size, 0, 0, pos_d, p->vocab_size, dim);
    // for(int b=0 ; b<batch_size ; ++b)    
    //     output_logits[b] = s_batch->logits + b * p->vocab_size;
        
    CHECK_HIP(hipDeviceSynchronize());
    CHECK_HIP(hipFree(pos_d));
    return thablas_status;
}

thablasStatus_t thaDNN_s_forward_batch_multiple_pipe_line(thablasHandle_t handle[], int fid, int n_flows, int n_devices, int batch_size, 
                                                          Config* p, TransformerWeights* w[], RunState* s_batch[], int token[], int pos[], 
                                                          float* logits_host, int* flow_status, int* device_flow, std::mutex *device_mtx) {
    int dim = p->dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads; 
    int kv_mul = p->n_heads / p->n_kv_heads; // integer multiplier of the kv sharing in multiquery
    int hidden_dim =  p->hidden_dim;
    int head_size = dim / p->n_heads;
    int pipe_size = p->n_layers / n_devices;

    thablasStatus_t thablas_status = THABLAS_STATUS_SUCCESS;

    // copy the token embedding into x
    device_mtx[0].lock();
    CHECK_HIP(hipSetDevice(0));
    float* content_row[batch_size];
    for(int b=0 ; b<batch_size ; ++b)
    {
        content_row[b] = w[0]->token_embedding_table + token[b] * dim;
        // memcpy(x[b], content_row[b], dim*sizeof(float)); // TODO: copy device to device
        CHECK_HIP(hipMemcpy(s_batch[0]->x + b * dim, content_row[b], dim * sizeof(float), hipMemcpyDeviceToDevice));
    }
    device_mtx[0].unlock();

    for(int gid=0; gid<n_devices ; ++gid) { // loop each device 
        device_mtx[gid].lock();
        CHECK_HIP(hipSetDevice(gid));
        int *pos_d;
        CHECK_HIP(hipMalloc(&pos_d, batch_size * sizeof(int)));
        CHECK_HIP(hipMemcpy(pos_d, pos, batch_size * sizeof(int), hipMemcpyHostToDevice));
        // forward all the layers
        for(unsigned long long l = 0; l < pipe_size; l++) {
            thablas_status = thaDNN_s_rmsnorm_v2_batch(handle[gid], batch_size, s_batch[gid]->xb, s_batch[gid]->x, w[gid]->rms_att_weight + l*dim, dim, dim);

            int loff = l * p->seq_len * kv_dim;

            thablas_status = thaBLAS_s_matmul_batch(handle[gid], batch_size, s_batch[gid]->q, s_batch[gid]->xb, w[gid]->wq + l*dim*dim, dim, dim, 0, 0, pos_d, dim, dim);
            thablas_status = thaBLAS_s_matmul_batch(handle[gid], batch_size, s_batch[gid]->key_cache, s_batch[gid]->xb, w[gid]->wk + l*dim*kv_dim, dim, kv_dim, loff, kv_dim, pos_d, pipe_size * p->seq_len * kv_dim, dim);
            thablas_status = thaBLAS_s_matmul_batch(handle[gid], batch_size, s_batch[gid]->value_cache, s_batch[gid]->xb, w[gid]->wv + l*dim*kv_dim, dim, kv_dim, loff, kv_dim, pos_d, pipe_size * p->seq_len * kv_dim, dim);

            for(int b=0 ; b<batch_size ; ++b)
                thablas_status = thaDNN_s_rope(handle[gid], dim, head_size, kv_dim, pos[b], s_batch[gid]->q + b * dim, s_batch[gid]->key_cache + loff + pos[b] * kv_dim + b * pipe_size * p->seq_len * kv_dim);

            // multi-head attention
            thablas_status = thaDNN_s_multiheads_1_v2_batch(handle[gid], batch_size, pos, pos_d, p->n_heads, pipe_size, s_batch[gid]->q, s_batch[gid]->att, s_batch[gid]->key_cache, head_size, p->seq_len, loff, kv_dim, dim, kv_mul);
            thablas_status = thaDNN_s_multiheads_2_batch(handle[gid], batch_size, s_batch[gid]->att, pos_d, p->seq_len, p->n_heads);
            thablas_status = thaDNN_s_multiheads_3_v2_batch(handle[gid], batch_size, pos_d, p->n_heads, s_batch[gid]->xb, s_batch[gid]->att, s_batch[gid]->value_cache, head_size, p->seq_len, loff, kv_dim, kv_mul, dim, pipe_size);

            thablas_status = thaBLAS_s_matmul_batch(handle[gid], batch_size, s_batch[gid]->xb2, s_batch[gid]->xb, w[gid]->wo + l*dim*dim, dim, dim, 0, 0, pos_d, dim, dim);

            for(int b=0 ; b<batch_size ; ++b)
                thablas_status = thaBLAS_s_vecaddvec(handle[gid], s_batch[gid]->x + b * dim, s_batch[gid]->xb2 + b * dim, dim);

            thablas_status = thaDNN_s_rmsnorm_v2_batch(handle[gid], batch_size, s_batch[gid]->xb, s_batch[gid]->x, w[gid]->rms_ffn_weight + l*dim, dim, dim);

            thablas_status = thaBLAS_s_matmul_batch(handle[gid], batch_size, s_batch[gid]->hb, s_batch[gid]->xb, w[gid]->w1 + l*dim*hidden_dim, dim, hidden_dim, 0, 0, pos_d, hidden_dim, dim);
            thablas_status = thaBLAS_s_matmul_batch(handle[gid], batch_size, s_batch[gid]->hb2, s_batch[gid]->xb, w[gid]->w3 + l*dim*hidden_dim, dim, hidden_dim, 0, 0, pos_d, hidden_dim, dim);

            for(int b=0 ; b<batch_size ; ++b)
                thablas_status = thaDNN_s_swiglu(handle[gid], s_batch[gid]->hb + b * hidden_dim, s_batch[gid]->hb2 + b * hidden_dim, hidden_dim);

            thablas_status = thaBLAS_s_matmul_batch(handle[gid], batch_size, s_batch[gid]->xb, s_batch[gid]->hb, w[gid]->w2 + l*dim*hidden_dim, hidden_dim, dim, 0, 0, pos_d, dim, hidden_dim);

            for(int b=0 ; b<batch_size ; ++b)
                thablas_status = thaBLAS_s_vecaddvec(handle[gid], s_batch[gid]->x + b * dim, s_batch[gid]->xb + b * dim, dim);
        }

        int next_device = (gid + 1) % n_devices;
        CHECK_HIP(hipMemcpy(s_batch[next_device]->x, s_batch[gid]->x, batch_size * dim * sizeof(float), hipMemcpyDeviceToDevice));
        CHECK_HIP(hipDeviceSynchronize());
        CHECK_HIP(hipFree(pos_d));
        device_mtx[gid].unlock();
    }

    device_mtx[0].lock();
    CHECK_HIP(hipSetDevice(0));
    int *pos_d;
    CHECK_HIP(hipMalloc(&pos_d, batch_size * sizeof(int)));
    CHECK_HIP(hipMemcpy(pos_d, pos, batch_size * sizeof(int), hipMemcpyHostToDevice));

    thablas_status = thaDNN_s_rmsnorm_v2_batch(handle[0], batch_size, s_batch[0]->x, s_batch[0]->x, w[0]->rms_final_weight, dim, dim);
    thablas_status = thaBLAS_s_matmul_batch(handle[0], batch_size, logits_host, s_batch[0]->x, w[0]->wcls, dim, p->vocab_size, 0, 0, pos_d, p->vocab_size, dim);
    // for(int b=0 ; b<batch_size ; ++b)    
    //     output_logits[b] = s_batch->logits + b * p->vocab_size;
        
    CHECK_HIP(hipDeviceSynchronize());
    CHECK_HIP(hipFree(pos_d));
    device_mtx[0].unlock();
    return thablas_status;
}
