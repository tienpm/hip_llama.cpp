#include "thaDNN.hpp"
#include "thaBLAS.hpp"
#include "hip_helper.hpp"
#include "seq.hpp"
#include "BatchManager.hpp"

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <vector>
#include <omp.h>


thablasStatus_t thaDNN_s_forward_batch(thablasHandle_t handle1, thablasHandle_t handle2, thablasHandle_t handle3, int batch_size, Config *p, TransformerWeights* w, 
                                       RunState* s_batch, int token[], int pos[], float* logits_host) {
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
    for (int b = 0 ; b < batch_size ; ++b) {
        content_row[b] = w->token_embedding_table + token[b] * dim;
        // memcpy(x[b], content_row[b], dim*sizeof(float)); // TODO: copy device to device
        CHECK_HIP(hipMemcpy(s_batch->x + b * dim, content_row[b], dim * sizeof(float), hipMemcpyDeviceToDevice));
    }

    // forward all the layers
    for(unsigned long long l = 0; l < p->n_layers; l++) {
        thablas_status = thaDNN_s_rmsnorm_v2_batch(&handle1, batch_size, s_batch->xb, s_batch->x, w->rms_att_weight + l * dim, dim, dim);

        int loff = l * p->seq_len * kv_dim;

        thablas_status = thaBLAS_s_matmul_batch(&handle1, batch_size, s_batch->q, s_batch->xb, w->wq + l*dim*dim, dim, dim, 0, 0, pos_d, dim, dim);
        thablas_status = thaBLAS_s_matmul_batch(&handle1, batch_size, s_batch->key_cache, s_batch->xb, w->wk + l*dim*kv_dim, dim, kv_dim, loff, kv_dim, pos_d, p->n_layers * p->seq_len * kv_dim, dim);
        thablas_status = thaBLAS_s_matmul_batch(&handle1, batch_size, s_batch->value_cache, s_batch->xb, w->wv + l*dim*kv_dim, dim, kv_dim, loff, kv_dim, pos_d, p->n_layers * p->seq_len * kv_dim, dim);

        for(int b=0 ; b<batch_size ; ++b)
            thablas_status = thaDNN_s_rope(&handle1, dim, head_size, kv_dim, pos[b], s_batch->q + b * dim, s_batch->key_cache + loff + pos[b] * kv_dim + b * p->n_layers * p->seq_len * kv_dim);

        // multi-head attention
        thablas_status = thaDNN_s_multiheads_1_v1_batch(&handle1, batch_size, pos, pos_d, p->n_heads, p->n_layers, s_batch->q, s_batch->att, s_batch->key_cache, head_size, p->seq_len, loff, kv_dim, dim, kv_mul);
        thablas_status = thaDNN_s_multiheads_2_v1_batch(&handle1, batch_size, s_batch->att, pos_d, p->seq_len, p->n_heads);
        thablas_status = thaDNN_s_multiheads_3_v1_batch(&handle1, batch_size, pos_d, p->n_heads, s_batch->xb, s_batch->att, s_batch->value_cache, head_size, p->seq_len, loff, kv_dim, kv_mul, dim, p->n_layers);

        thablas_status = thaBLAS_s_matmul_batch(&handle1, batch_size, s_batch->xb2, s_batch->xb, w->wo + l*dim*dim, dim, dim, 0, 0, pos_d, dim, dim);

        for(int b=0 ; b<batch_size ; ++b)
            thablas_status = thaBLAS_s_vecaddvec(&handle1, x[b], s_batch->xb2 + b * dim, dim);

        thablas_status = thaDNN_s_rmsnorm_v2_batch(&handle1, batch_size, s_batch->xb, s_batch->x, w->rms_ffn_weight + l*dim, dim, dim);

        thablas_status = thaBLAS_s_matmul_batch(&handle1, batch_size, s_batch->hb, s_batch->xb, w->w1 + l*dim*hidden_dim, dim, hidden_dim, 0, 0, pos_d, hidden_dim, dim);
        thablas_status = thaBLAS_s_matmul_batch(&handle1, batch_size, s_batch->hb2, s_batch->xb, w->w3 + l*dim*hidden_dim, dim, hidden_dim, 0, 0, pos_d, hidden_dim, dim);

        for(int b=0 ; b<batch_size ; ++b)
            thablas_status = thaDNN_s_swiglu(&handle1, s_batch->hb + b * hidden_dim, s_batch->hb2 + b * hidden_dim, hidden_dim);

        thablas_status = thaBLAS_s_matmul_batch(&handle1, batch_size, s_batch->xb, s_batch->hb, w->w2 + l*dim*hidden_dim, hidden_dim, dim, 0, 0, pos_d, dim, hidden_dim);

        for(int b=0 ; b<batch_size ; ++b)
            thablas_status = thaBLAS_s_vecaddvec(&handle1, x[b], s_batch->xb + b * dim, dim);
    }

    thablas_status = thaDNN_s_rmsnorm_v2_batch(&handle1, batch_size, s_batch->x, s_batch->x, w->rms_final_weight, dim, dim);
    thablas_status = thaBLAS_s_matmul_batch(&handle1, batch_size, logits_host, s_batch->x, w->wcls, dim, p->vocab_size, 0, 0, pos_d, p->vocab_size, dim);
        
    CHECK_HIP(hipDeviceSynchronize());
    CHECK_HIP(hipFree(pos_d));
    return thablas_status;
}

thablasStatus_t thaDNN_s_forward_70B(thablasHandle_t handle, int batch_size, Config *p, TransformerWeights* h_w[], RunState* h_s, TransformerWeights* d_w, RunState* d_s, int token[], int pos[], float* logits_host) {
    float *x[batch_size];
    int dim = p->dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads; 
    int kv_mul = p->n_heads / p->n_kv_heads; // integer multiplier of the kv sharing in multiquery
    int hidden_dim =  p->hidden_dim;
    int head_size = dim / p->n_heads;
    for(int b=0 ; b<batch_size ; ++b)
        x[b] = d_s->x + b * dim;

    int *pos_d;
    CHECK_HIP(hipMalloc(&pos_d, batch_size * sizeof(int)));
    CHECK_HIP(hipMemcpy(pos_d, pos, batch_size * sizeof(int), hipMemcpyHostToDevice));

    thablasStatus_t thablas_status = THABLAS_STATUS_SUCCESS;

    // fprintf(stderr, "CP0\n");

    // copy the token embedding into x
    float* content_row[batch_size];
    for(int b=0 ; b<batch_size ; ++b)
    {
        content_row[b] = d_w->token_embedding_table + token[b] * dim;
        // memcpy(x[b], content_row[b], dim*sizeof(float)); // TODO: copy device to device
        CHECK_HIP(hipMemcpy(d_s->x + b * dim, content_row[b], dim*sizeof(float), hipMemcpyDeviceToDevice));
    }

    // forward all the layers
    for(unsigned long long l = 0; l < p->n_layers; l++) {

        // CHECK_HIP(hipMemcpy(d_w->rms_att_weight, h_w->rms_att_weight + l*dim, 1 * dim * sizeof(float), hipMemcpyHostToDevice));
        // CHECK_HIP(hipMemcpy(d_w->rms_ffn_weight, h_w->rms_ffn_weight + l*dim, 1 * dim * sizeof(float), hipMemcpyHostToDevice));
        // CHECK_HIP(hipMemcpy(d_w->wq,             h_w->wq + l*dim*dim, 1 * dim * dim * sizeof(float), hipMemcpyHostToDevice));
        // CHECK_HIP(hipMemcpy(d_w->wk,             h_w->wk + l*dim*kv_dim, 1 * dim * kv_dim * sizeof(float), hipMemcpyHostToDevice));
        // CHECK_HIP(hipMemcpy(d_w->wv,             h_w->wv + l*dim*kv_dim, 1 * dim * kv_dim * sizeof(float), hipMemcpyHostToDevice));
        // CHECK_HIP(hipMemcpy(d_w->wo,             h_w->wo + l*dim*dim, 1 * dim * dim * sizeof(float), hipMemcpyHostToDevice));
        // CHECK_HIP(hipMemcpy(d_w->w1,             h_w->w1 + l*hidden_dim*dim, 1 * hidden_dim * dim * sizeof(float), hipMemcpyHostToDevice));
        // CHECK_HIP(hipMemcpy(d_w->w2,             h_w->w2 + l*dim*hidden_dim, 1 * dim * hidden_dim * sizeof(float), hipMemcpyHostToDevice));
        // CHECK_HIP(hipMemcpy(d_w->w3,             h_w->w3 + l*hidden_dim*dim, 1 * hidden_dim * dim * sizeof(float), hipMemcpyHostToDevice));

        CHECK_HIP(hipMemcpy(d_w->rms_att_weight, h_w[l]->rms_att_weight, 1 * dim * sizeof(float), hipMemcpyHostToDevice));
        CHECK_HIP(hipMemcpy(d_w->rms_ffn_weight, h_w[l]->rms_ffn_weight, 1 * dim * sizeof(float), hipMemcpyHostToDevice));
        CHECK_HIP(hipMemcpy(d_w->wq,             h_w[l]->wq, 1 * dim * dim * sizeof(float), hipMemcpyHostToDevice));
        CHECK_HIP(hipMemcpy(d_w->wk,             h_w[l]->wk, 1 * dim * kv_dim * sizeof(float), hipMemcpyHostToDevice));
        CHECK_HIP(hipMemcpy(d_w->wv,             h_w[l]->wv, 1 * dim * kv_dim * sizeof(float), hipMemcpyHostToDevice));
        CHECK_HIP(hipMemcpy(d_w->wo,             h_w[l]->wo, 1 * dim * dim * sizeof(float), hipMemcpyHostToDevice));
        CHECK_HIP(hipMemcpy(d_w->w1,             h_w[l]->w1, 1 * hidden_dim * dim * sizeof(float), hipMemcpyHostToDevice));
        CHECK_HIP(hipMemcpy(d_w->w2,             h_w[l]->w2, 1 * dim * hidden_dim * sizeof(float), hipMemcpyHostToDevice));
        CHECK_HIP(hipMemcpy(d_w->w3,             h_w[l]->w3, 1 * hidden_dim * dim * sizeof(float), hipMemcpyHostToDevice));

        CHECK_HIP(hipMemcpy(d_s->key_cache,   h_s->key_cache   + l * p->seq_len * kv_dim, (pos[0]+1) * kv_dim * sizeof(float), hipMemcpyHostToDevice));
        CHECK_HIP(hipMemcpy(d_s->value_cache, h_s->value_cache + l * p->seq_len * kv_dim, (pos[0]+1) * kv_dim * sizeof(float), hipMemcpyHostToDevice));
        CHECK_HIP(hipDeviceSynchronize());
        
        // fprintf(stderr, "CP1\n");

        thablas_status = thaDNN_s_rmsnorm_v2_batch(&handle, batch_size, d_s->xb, d_s->x, d_w->rms_att_weight, dim, dim);

        int loff = 0;

        thablas_status = thaBLAS_s_matmul_batch(&handle, batch_size, d_s->q, d_s->xb, d_w->wq, dim, dim, 0, 0, pos_d, dim, dim);
        thablas_status = thaBLAS_s_matmul_batch(&handle, batch_size, d_s->key_cache, d_s->xb, d_w->wk, dim, kv_dim, loff, kv_dim, pos_d, p->n_layers * p->seq_len * kv_dim, dim);
        thablas_status = thaBLAS_s_matmul_batch(&handle, batch_size, d_s->value_cache, d_s->xb, d_w->wv, dim, kv_dim, loff, kv_dim, pos_d, p->n_layers * p->seq_len * kv_dim, dim);

        // fprintf(stderr, "CP2\n");

        for(int b=0 ; b<batch_size ; ++b)
            thablas_status = thaDNN_s_rope(&handle, dim, head_size, kv_dim, pos[b], d_s->q + b * dim, d_s->key_cache + loff + pos[b] * kv_dim + b * p->n_layers * p->seq_len * kv_dim);

        // multi-head attention
        thablas_status = thaDNN_s_multiheads_1_v1_batch(&handle, batch_size, pos, pos_d, p->n_heads, p->n_layers, d_s->q, d_s->att, d_s->key_cache, head_size, p->seq_len, loff, kv_dim, dim, kv_mul);
        thablas_status = thaDNN_s_multiheads_2_v1_batch(&handle, batch_size, d_s->att, pos_d, p->seq_len, p->n_heads);
        thablas_status = thaDNN_s_multiheads_3_v1_batch(&handle, batch_size, pos_d, p->n_heads, d_s->xb, d_s->att, d_s->value_cache, head_size, p->seq_len, loff, kv_dim, kv_mul, dim, p->n_layers);

        thablas_status = thaBLAS_s_matmul_batch(&handle, batch_size, d_s->xb2, d_s->xb, d_w->wo, dim, dim, 0, 0, pos_d, dim, dim);

        for(int b=0 ; b<batch_size ; ++b)
            thablas_status = thaBLAS_s_vecaddvec(&handle, x[b], d_s->xb2 + b * dim, dim);

        thablas_status = thaDNN_s_rmsnorm_v2_batch(&handle, batch_size, d_s->xb, d_s->x, d_w->rms_ffn_weight, dim, dim);

        thablas_status = thaBLAS_s_matmul_batch(&handle, batch_size, d_s->hb, d_s->xb, d_w->w1, dim, hidden_dim, 0, 0, pos_d, hidden_dim, dim);
        thablas_status = thaBLAS_s_matmul_batch(&handle, batch_size, d_s->hb2, d_s->xb, d_w->w3, dim, hidden_dim, 0, 0, pos_d, hidden_dim, dim);

        for(int b=0 ; b<batch_size ; ++b)
            thablas_status = thaDNN_s_swiglu(&handle, d_s->hb + b * hidden_dim, d_s->hb2 + b * hidden_dim, hidden_dim);

        thablas_status = thaBLAS_s_matmul_batch(&handle, batch_size, d_s->xb, d_s->hb, d_w->w2, hidden_dim, dim, 0, 0, pos_d, dim, hidden_dim);

        for(int b=0 ; b<batch_size ; ++b)
            thablas_status = thaBLAS_s_vecaddvec(&handle, x[b], d_s->xb + b * dim, dim);

        CHECK_HIP(hipDeviceSynchronize());
        CHECK_HIP(hipMemcpy(h_s->key_cache   + l * p->seq_len * kv_dim, d_s->key_cache,   (pos[0]+1) * kv_dim * sizeof(float), hipMemcpyDeviceToHost));
        CHECK_HIP(hipMemcpy(h_s->value_cache + l * p->seq_len * kv_dim, d_s->value_cache, (pos[0]+1) * kv_dim * sizeof(float), hipMemcpyDeviceToHost));
        CHECK_HIP(hipDeviceSynchronize());
    }

    thablas_status = thaDNN_s_rmsnorm_v2_batch(&handle, batch_size, d_s->x, d_s->x, d_w->rms_final_weight, dim, dim);
    thablas_status = thaBLAS_s_matmul_batch(&handle, batch_size, logits_host, d_s->x, d_w->wcls, dim, p->vocab_size, 0, 0, pos_d, p->vocab_size, dim);
    // for(int b=0 ; b<batch_size ; ++b)    
    //     output_logits[b] = d_s->logits + b * p->vocab_size;
        
    CHECK_HIP(hipDeviceSynchronize());
    CHECK_HIP(hipFree(pos_d));
    return thablas_status;
}

thablasStatus_t thaDNN_s_forward_batch_multiple_pipe_line(thablasHandle_t handle[], int fid, int n_host_threads, int n_devices, int batch_size, 
                                                          Config* p, TransformerWeights* w[], RunState* s_batch[], int token[], int pos[], 
                                                          float* logits_host, int* host_thread_status, int* device_host_thread, omp_lock_t *device_locks) {
    int dim = p->dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads; 
    int kv_mul = p->n_heads / p->n_kv_heads; // integer multiplier of the kv sharing in multiquery
    int hidden_dim =  p->hidden_dim;
    int head_size = dim / p->n_heads;
    int pipe_size = p->n_layers / n_devices;
    thablasStatus_t thablas_status = THABLAS_STATUS_SUCCESS;
    float* content_row[batch_size];

    for(int gid=0; gid<n_devices ; ++gid) { // loop each device 
        omp_set_lock(&device_locks[gid]);
        CHECK_HIP(hipSetDevice(gid));

        if (gid == 0) {
            // copy the token embedding into x
            for(int b=0 ; b<batch_size ; ++b) {
                content_row[b] = w[0]->token_embedding_table + token[b] * dim;
                CHECK_HIP(hipMemcpy(s_batch[0]->x + b * dim, content_row[b], dim * sizeof(float), hipMemcpyDeviceToDevice));
            }
        }

        int *pos_d;
        int max_pos = pos[0];
        for(int b=1 ; b<batch_size ; ++b) max_pos = std::max(pos[b], max_pos);
        CHECK_HIP(hipMalloc(&pos_d, batch_size * sizeof(int)));
        CHECK_HIP(hipMemcpy(pos_d, pos, batch_size * sizeof(int), hipMemcpyHostToDevice));

        // forward all the layers
        for(unsigned long long l = 0; l < pipe_size; l++) {
        
            int loff = l * p->seq_len * batch_size * kv_dim;
            float* s_batch_key_layer_cache = s_batch[gid]->key_cache + loff;
            float* s_batch_value_layer_cache = s_batch[gid]->value_cache + loff;

            thablas_status = thaDNN_s_rmsnorm_v2_batch(&handle[gid], batch_size, s_batch[gid]->xb, s_batch[gid]->x, w[gid]->rms_att_weight + l*dim, dim, dim);

            thablas_status = thaBLAS_s_matmul_ifdef(&handle[gid], w[gid]->wq + l*dim*dim, s_batch[gid]->xb, s_batch[gid]->q,   dim, batch_size, dim);
            thablas_status = thaBLAS_s_matmul_ifdef(&handle[gid], w[gid]->wk + l*dim*kv_dim, s_batch[gid]->xb, s_batch[gid]->key_matmul,   kv_dim, batch_size, dim);
            thablas_status = thaBLAS_s_matmul_ifdef(&handle[gid], w[gid]->wv + l*dim*kv_dim, s_batch[gid]->xb, s_batch[gid]->value_matmul, kv_dim, batch_size, dim);
            
            for(int b=0 ; b<batch_size ; ++b) {
                int offset = pos[b] * batch_size * kv_dim + b * kv_dim;
                // TODO: use kernel for copy
                CHECK_HIP(hipMemcpy(s_batch_key_layer_cache   + offset, s_batch[gid]->key_matmul   + b * kv_dim, kv_dim * sizeof(float), hipMemcpyDeviceToDevice));
                CHECK_HIP(hipMemcpy(s_batch_value_layer_cache + offset, s_batch[gid]->value_matmul + b * kv_dim, kv_dim * sizeof(float), hipMemcpyDeviceToDevice));
                thablas_status = thaDNN_s_rope(&handle[gid], dim, head_size, kv_dim, pos[b], s_batch[gid]->q + b * dim, s_batch_key_layer_cache + offset);
            }

            // multi-head attention
            int multi_head_n_words = p->seq_len;
            thablas_status = thaDNN_s_multiheads_1_v2_batch(&handle[gid], batch_size, pipe_size, pos, pos_d, p->n_heads, s_batch[gid]->q, s_batch[gid]->att, s_batch_key_layer_cache, head_size, multi_head_n_words, kv_dim, dim, kv_mul);
            thablas_status = thaDNN_s_multiheads_2_batch(&handle[gid], batch_size, s_batch[gid]->att, pos_d, multi_head_n_words, p->n_heads);
            thablas_status = thaDNN_s_multiheads_3_v2_batch(&handle[gid], batch_size, pos_d, p->n_heads, s_batch[gid]->xb, s_batch[gid]->att, s_batch_value_layer_cache, head_size, multi_head_n_words, kv_dim, kv_mul, dim, pipe_size);

            thablas_status = thaBLAS_s_matmul_ifdef(&handle[gid], w[gid]->wo + l*dim*dim, s_batch[gid]->xb, s_batch[gid]->xb2, dim, batch_size, dim);

            for(int b=0 ; b<batch_size ; ++b)
                thablas_status = thaBLAS_s_vecaddvec(&handle[gid], s_batch[gid]->x + b * dim, s_batch[gid]->xb2 + b * dim, dim);

            thablas_status = thaDNN_s_rmsnorm_v2_batch(&handle[gid], batch_size, s_batch[gid]->xb, s_batch[gid]->x, w[gid]->rms_ffn_weight + l*dim, dim, dim);

            thablas_status = thaBLAS_s_matmul_ifdef(&handle[gid], w[gid]->w1 + l*dim*hidden_dim, s_batch[gid]->xb, s_batch[gid]->hb, hidden_dim, batch_size, dim);
            thablas_status = thaBLAS_s_matmul_ifdef(&handle[gid], w[gid]->w3 + l*dim*hidden_dim, s_batch[gid]->xb, s_batch[gid]->hb2, hidden_dim, batch_size, dim);

            for(int b=0 ; b<batch_size ; ++b)
                thablas_status = thaDNN_s_swiglu(&handle[gid], s_batch[gid]->hb + b * hidden_dim, s_batch[gid]->hb2 + b * hidden_dim, hidden_dim);

            thablas_status = thaBLAS_s_matmul_ifdef(&handle[gid], w[gid]->w2 + l*dim*hidden_dim, s_batch[gid]->hb, s_batch[gid]->xb, dim, batch_size, hidden_dim);

            for(int b=0 ; b<batch_size ; ++b)
                thablas_status = thaBLAS_s_vecaddvec(&handle[gid], s_batch[gid]->x + b * dim, s_batch[gid]->xb + b * dim, dim);
        }

        int next_device = gid + 1;
        if (next_device < n_devices) {
            CHECK_HIP(hipMemcpy(s_batch[next_device]->x, s_batch[gid]->x, batch_size * dim * sizeof(float), hipMemcpyDeviceToDevice));
            // CHECK_HIP(hipDeviceSynchronize());
        } else {
            thablas_status = thaDNN_s_rmsnorm_v2_batch(&handle[gid], batch_size, s_batch[gid]->x, s_batch[gid]->x, w[gid]->rms_final_weight, dim, dim);
            thablas_status = thaBLAS_s_matmul_ifdef(&handle[gid], w[gid]->wcls, s_batch[gid]->x, logits_host, p->vocab_size, batch_size, dim);
        }

        CHECK_HIP(hipFree(pos_d));
        CHECK_HIP(hipDeviceSynchronize());
        omp_unset_lock(&device_locks[gid]);
    }

    return thablas_status;
}


// Forward using layer swap
// In each layer:
// 1. Copy kv_cache data from host to device kv_layer_cache while calculating the current kv_cache
// 2. Move data from calculated kv_cache to kv_layer_cache
// 3. Use kv_layer_cache to calculate multi-head attention
// 4. Copy kv_layer_cache back to host kv_cache
thablasStatus_t thaDNN_s_forward_batch_multiple_pipe_line_layer_swap(thablasHandle_t handle[], int thread_id, int n_host_threads, int n_devices, int batch_size, int n_buffer_words, 
                                                                     Config* p, TransformerWeights* w[], RunState* s_batch[], RunState* s_host_batch[], int token[], int pos[], 
                                                                     float* logits_host, omp_lock_t *device_locks) {

    int dim = p->dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads; 
    int kv_mul = p->n_heads / p->n_kv_heads; // integer multiplier of the kv sharing in multiquery
    int hidden_dim =  p->hidden_dim;
    int head_size = dim / p->n_heads;
    int pipe_size = p->n_layers / n_devices;
    thablasStatus_t thablas_status = THABLAS_STATUS_SUCCESS;

    for(int gid=0; gid<n_devices ; ++gid) { // loop each device 
        omp_set_lock(&device_locks[gid]);
        CHECK_HIP(hipSetDevice(gid));

        // hipEvent_t copy_done_event[pipe_size];
        // for(unsigned long long l = 0; l < pipe_size; l++)
        //     CHECK_HIP(hipEventCreate(&copy_done_event[l]));
        // CHECK_HIP(hipDeviceSynchronize());

        if (gid == 0) {
            // copy the token embedding into x
            float* content_row[batch_size];
            for(int b=0 ; b<batch_size ; ++b) {
                content_row[b] = w[0]->token_embedding_table + token[b] * dim;
                CHECK_HIP(hipMemcpyAsync(s_batch[0]->x + b * dim, content_row[b], dim * sizeof(float), hipMemcpyDeviceToDevice, handle[gid].calc_stream));
            }
        }

        int *pos_d;
        int max_pos = pos[0];
        for(int b=1 ; b<batch_size ; ++b) max_pos = std::max(pos[b], max_pos);
        CHECK_HIP(hipMallocAsync(&pos_d, batch_size * sizeof(int), handle[gid].copy_stream));
        CHECK_HIP(hipMemcpyAsync(pos_d, pos, batch_size * sizeof(int), hipMemcpyHostToDevice, handle[gid].copy_stream));

        // forward all the layers
        for(unsigned long long l = 0; l < pipe_size; l++) {
        
            thablas_status = thaDNN_s_rmsnorm_v2_batch(&handle[gid], batch_size, s_batch[gid]->xb, s_batch[gid]->x, w[gid]->rms_att_weight + l*dim, dim, dim);
            thablas_status = thaBLAS_s_matmul_ifdef(&handle[gid], w[gid]->wq + l*dim*dim,    s_batch[gid]->xb, s_batch[gid]->q,            dim,    batch_size, dim);
            thablas_status = thaBLAS_s_matmul_ifdef(&handle[gid], w[gid]->wk + l*dim*kv_dim, s_batch[gid]->xb, s_batch[gid]->key_matmul,   kv_dim, batch_size, dim);
            thablas_status = thaBLAS_s_matmul_ifdef(&handle[gid], w[gid]->wv + l*dim*kv_dim, s_batch[gid]->xb, s_batch[gid]->value_matmul, kv_dim, batch_size, dim);

            float* s_batch_key_layer_cache;
            float* s_batch_value_layer_cache;
            int loff = l * n_buffer_words * batch_size * kv_dim;
            if (max_pos + 1 <= n_buffer_words) {
                s_batch_key_layer_cache = s_batch[gid]->key_cache + loff;
                s_batch_value_layer_cache = s_batch[gid]->value_cache + loff;
            } else 
            {
                // 1. Copy kv_cache data from host to device kv_layer_cache while calculating the current kv_cache
                int total_words_to_swap = max_pos + 1 - n_buffer_words;
                int total_buffer_words = n_buffer_words * batch_size * kv_dim;

                s_batch_key_layer_cache = s_batch[gid]->key_layer_cache;
                s_batch_value_layer_cache = s_batch[gid]->value_layer_cache;
                CHECK_HIP(hipMemcpyAsync(s_batch_key_layer_cache, s_batch[gid]->key_cache + loff, total_buffer_words * sizeof(float), hipMemcpyDeviceToDevice, handle[gid].copy_stream));
                CHECK_HIP(hipMemcpyAsync(s_batch_value_layer_cache, s_batch[gid]->value_cache + loff, total_buffer_words * sizeof(float), hipMemcpyDeviceToDevice, handle[gid].copy_stream));

                int host_loff = l * (p->seq_len - n_buffer_words) * batch_size * kv_dim;
                CHECK_HIP(hipMemcpyAsync(s_batch_key_layer_cache + total_buffer_words, s_host_batch[gid]->key_cache + host_loff, total_words_to_swap * batch_size * kv_dim * sizeof(float), hipMemcpyHostToDevice, handle[gid].copy_stream));
                CHECK_HIP(hipMemcpyAsync(s_batch_value_layer_cache + total_buffer_words, s_host_batch[gid]->value_cache + host_loff, total_words_to_swap * batch_size * kv_dim * sizeof(float), hipMemcpyHostToDevice, handle[gid].copy_stream));
            }
            // CHECK_HIP(hipEventRecord(copy_done_event[l], handle->copy_stream));
            // CHECK_HIP(hipStreamWaitEvent(handle->calc_stream, copy_done_event[l], 0));

            // 2. Move data from calculated kv_cache to kv_layer_cache
            // CHECK_HIP(hipStreamSynchronize(handle[gid].copy_stream));
            CHECK_HIP(hipDeviceSynchronize());
            for(int b=0 ; b<batch_size ; ++b) {
                int offset = pos[b] * batch_size * kv_dim + b * kv_dim;
                // TODO: use kernel for copy
                CHECK_HIP(hipMemcpyAsync(s_batch_key_layer_cache   + offset, s_batch[gid]->key_matmul   + b * kv_dim, kv_dim * sizeof(float), hipMemcpyDeviceToDevice, handle[gid].calc_stream));
                CHECK_HIP(hipMemcpyAsync(s_batch_value_layer_cache + offset, s_batch[gid]->value_matmul + b * kv_dim, kv_dim * sizeof(float), hipMemcpyDeviceToDevice, handle[gid].calc_stream));
                thablas_status = thaDNN_s_rope(&handle[gid], dim, head_size, kv_dim, pos[b], s_batch[gid]->q + b * dim, s_batch_key_layer_cache + offset);
            }
            CHECK_HIP(hipDeviceSynchronize());

            // 3. Use kv_layer_cache to calculate multi-head attention
            int multi_head_n_buffer_words = p->seq_len;
            thablas_status = thaDNN_s_multiheads_1_v2_batch(&handle[gid], batch_size, pipe_size, pos, pos_d, p->n_heads, s_batch[gid]->q, s_batch[gid]->att, s_batch_key_layer_cache, head_size, multi_head_n_buffer_words, kv_dim, dim, kv_mul);
            thablas_status = thaDNN_s_multiheads_2_batch(&handle[gid], batch_size, s_batch[gid]->att, pos_d, multi_head_n_buffer_words, p->n_heads);
            thablas_status = thaDNN_s_multiheads_3_v2_batch(&handle[gid], batch_size, pos_d, p->n_heads, s_batch[gid]->xb, s_batch[gid]->att, s_batch_value_layer_cache, head_size, multi_head_n_buffer_words, kv_dim, kv_mul, dim, pipe_size);

            thablas_status = thaBLAS_s_matmul_ifdef(&handle[gid], w[gid]->wo + l*dim*dim, s_batch[gid]->xb, s_batch[gid]->xb2, dim, batch_size, dim);

            if (max_pos + 1 > n_buffer_words) 
            {
                // 4. Copy kv_layer_cache back to host kv_cache
                int total_words_to_swap = max_pos + 1 - n_buffer_words;
                int total_buffer_words = n_buffer_words * batch_size * kv_dim;

                CHECK_HIP(hipMemcpyAsync(s_batch[gid]->key_cache + loff, s_batch_key_layer_cache, total_buffer_words * sizeof(float), hipMemcpyDeviceToDevice, handle[gid].copy_stream));
                CHECK_HIP(hipMemcpyAsync(s_batch[gid]->value_cache + loff, s_batch_value_layer_cache, total_buffer_words * sizeof(float), hipMemcpyDeviceToDevice, handle[gid].copy_stream));

                int host_loff = l * (p->seq_len - n_buffer_words) * batch_size * kv_dim;
                CHECK_HIP(hipMemcpyAsync(s_host_batch[gid]->key_cache + host_loff, s_batch_key_layer_cache + total_buffer_words, total_words_to_swap * batch_size * kv_dim * sizeof(float), hipMemcpyDeviceToHost, handle[gid].copy_stream));
                CHECK_HIP(hipMemcpyAsync(s_host_batch[gid]->value_cache + host_loff, s_batch_value_layer_cache + total_buffer_words, total_words_to_swap * batch_size * kv_dim * sizeof(float), hipMemcpyDeviceToHost, handle[gid].copy_stream));
            }

            for(int b=0 ; b<batch_size ; ++b)
                thablas_status = thaBLAS_s_vecaddvec(&handle[gid], s_batch[gid]->x + b * dim, s_batch[gid]->xb2 + b * dim, dim);

            thablas_status = thaDNN_s_rmsnorm_v2_batch(&handle[gid], batch_size, s_batch[gid]->xb, s_batch[gid]->x, w[gid]->rms_ffn_weight + l*dim, dim, dim);

            thablas_status = thaBLAS_s_matmul_ifdef(&handle[gid], w[gid]->w1 + l*dim*hidden_dim, s_batch[gid]->xb, s_batch[gid]->hb, hidden_dim, batch_size, dim);
            thablas_status = thaBLAS_s_matmul_ifdef(&handle[gid], w[gid]->w3 + l*dim*hidden_dim, s_batch[gid]->xb, s_batch[gid]->hb2, hidden_dim, batch_size, dim);

            for(int b=0 ; b<batch_size ; ++b)
                thablas_status = thaDNN_s_swiglu(&handle[gid], s_batch[gid]->hb + b * hidden_dim, s_batch[gid]->hb2 + b * hidden_dim, hidden_dim);

            thablas_status = thaBLAS_s_matmul_ifdef(&handle[gid], w[gid]->w2 + l*dim*hidden_dim, s_batch[gid]->hb, s_batch[gid]->xb, dim, batch_size, hidden_dim);

            for(int b=0 ; b<batch_size ; ++b)
                thablas_status = thaBLAS_s_vecaddvec(&handle[gid], s_batch[gid]->x + b * dim, s_batch[gid]->xb + b * dim, dim);
            CHECK_HIP(hipDeviceSynchronize());
        }

        int next_device = gid + 1;
        if (next_device < n_devices) {
            CHECK_HIP(hipMemcpyAsync(s_batch[next_device]->x, s_batch[gid]->x, batch_size * dim * sizeof(float), hipMemcpyDeviceToDevice, handle[gid].calc_stream));
        } else {
            thablas_status = thaDNN_s_rmsnorm_v2_batch(&handle[gid], batch_size, s_batch[gid]->x, s_batch[gid]->x, w[gid]->rms_final_weight, dim, dim);
            thablas_status = thaBLAS_s_matmul_ifdef(&handle[gid], w[gid]->wcls, s_batch[gid]->x, logits_host, p->vocab_size, batch_size, dim);
        }
        
        CHECK_HIP(hipDeviceSynchronize());
        // for(unsigned long long l = 0; l < pipe_size; l++)
        //     CHECK_HIP(hipEventDestroy(copy_done_event[l]));
        CHECK_HIP(hipFree(pos_d));
        omp_unset_lock(&device_locks[gid]);
    }
     
    return thablas_status;
}


// thablasStatus_t thaDNN_s_forward_batch_multiple_pipe_line_paged_att(thablasHandle_t handle[], int thread_id, int n_host_threads, int n_devices, int batch_size, int n_buffer_words, 
//                                                                     Config* p, TransformerWeights* w[], RunState* s_batch[], std::vector<KVBlock> **kv_caches, BatchManager bm[], int token[], int pos[], 
//                                                                     float* logits_host, omp_lock_t *device_locks) {

//     int dim = p->dim;
//     int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads; 
//     int kv_mul = p->n_heads / p->n_kv_heads; // integer multiplier of the kv sharing in multiquery
//     int hidden_dim =  p->hidden_dim;
//     int head_size = dim / p->n_heads;
//     int pipe_size = p->n_layers / n_devices;
//     thablasStatus_t thablas_status = THABLAS_STATUS_SUCCESS;

//     for(int gid=0; gid<n_devices ; ++gid) { // loop each device 
//         omp_set_lock(&device_locks[gid]);
//         CHECK_HIP(hipSetDevice(gid));

//         // hipEvent_t copy_done_event[pipe_size];
//         // for(unsigned long long l = 0; l < pipe_size; l++)
//         //     CHECK_HIP(hipEventCreate(&copy_done_event[l]));
//         // CHECK_HIP(hipDeviceSynchronize());

//         if (gid == 0) {
//             // copy the token embedding into x
//             float* content_row[batch_size];
//             for(int b=0 ; b<batch_size ; ++b) {
//                 content_row[b] = w[0]->token_embedding_table + token[b] * dim;
//                 CHECK_HIP(hipMemcpyAsync(s_batch[0]->x + b * dim, content_row[b], dim * sizeof(float), hipMemcpyDeviceToDevice, handle[gid].calc_stream));
//             }
//         }

//         int *pos_d;
//         int max_pos = pos[0];
//         for(int b=1 ; b<batch_size ; ++b) max_pos = std::max(pos[b], max_pos);
//         CHECK_HIP(hipMallocAsync(&pos_d, batch_size * sizeof(int), handle[gid].copy_stream));
//         CHECK_HIP(hipMemcpyAsync(pos_d, pos, batch_size * sizeof(int), hipMemcpyHostToDevice, handle[gid].copy_stream));

//         // forward all the layers
//         for(unsigned long long l = 0; l < pipe_size; l++) {
        
//             thablas_status = thaDNN_s_rmsnorm_v2_batch(&handle[gid], batch_size, s_batch[gid]->xb, s_batch[gid]->x, w[gid]->rms_att_weight + l*dim, dim, dim);
//             thablas_status = thaBLAS_s_matmul_ifdef(&handle[gid], w[gid]->wq + l*dim*dim,    s_batch[gid]->xb, s_batch[gid]->q,            dim,    batch_size, dim);
//             thablas_status = thaBLAS_s_matmul_ifdef(&handle[gid], w[gid]->wk + l*dim*kv_dim, s_batch[gid]->xb, s_batch[gid]->key_matmul,   kv_dim, batch_size, dim);
//             thablas_status = thaBLAS_s_matmul_ifdef(&handle[gid], w[gid]->wv + l*dim*kv_dim, s_batch[gid]->xb, s_batch[gid]->value_matmul, kv_dim, batch_size, dim);

//             float* s_batch_key_layer_cache;
//             float* s_batch_value_layer_cache;
//             int loff = l * n_buffer_words * batch_size * kv_dim;
//             // if (max_pos + 1 <= n_buffer_words) {
//             //     s_batch_key_layer_cache = s_batch[gid]->key_cache + loff;
//             //     s_batch_value_layer_cache = s_batch[gid]->value_cache + loff;
//             // } else 
//             // {
//             //     // 1. Copy kv_cache data from host to device kv_layer_cache while calculating the current kv_cache
//             //     int total_words_to_swap = max_pos + 1 - n_buffer_words;
//             //     int total_buffer_words = n_buffer_words * batch_size * kv_dim;

//             //     s_batch_key_layer_cache = s_batch[gid]->key_layer_cache;
//             //     s_batch_value_layer_cache = s_batch[gid]->value_layer_cache;
//             //     CHECK_HIP(hipMemcpyAsync(s_batch_key_layer_cache, s_batch[gid]->key_cache + loff, total_buffer_words * sizeof(float), hipMemcpyDeviceToDevice, handle[gid].copy_stream));
//             //     CHECK_HIP(hipMemcpyAsync(s_batch_value_layer_cache, s_batch[gid]->value_cache + loff, total_buffer_words * sizeof(float), hipMemcpyDeviceToDevice, handle[gid].copy_stream));

//             //     int host_loff = l * (p->seq_len - n_buffer_words) * batch_size * kv_dim;
//             //     CHECK_HIP(hipMemcpyAsync(s_batch_key_layer_cache + total_buffer_words, s_host_batch[gid]->key_cache + host_loff, total_words_to_swap * batch_size * kv_dim * sizeof(float), hipMemcpyHostToDevice, handle[gid].copy_stream));
//             //     CHECK_HIP(hipMemcpyAsync(s_batch_value_layer_cache + total_buffer_words, s_host_batch[gid]->value_cache + host_loff, total_words_to_swap * batch_size * kv_dim * sizeof(float), hipMemcpyHostToDevice, handle[gid].copy_stream));
//             // }
//             // CHECK_HIP(hipEventRecord(copy_done_event[l], handle->copy_stream));
//             // CHECK_HIP(hipStreamWaitEvent(handle->calc_stream, copy_done_event[l], 0));

//             // 2. Move data from calculated kv_cache to kv_layer_cache
//             // CHECK_HIP(hipStreamSynchronize(handle[gid].copy_stream));
//             CHECK_HIP(hipDeviceSynchronize());
//             for(int b=0 ; b<batch_size ; ++b) {
//                 int offset = pos[b] * batch_size * kv_dim + b * kv_dim;
//                 // TODO: use kernel for copy
//                 // CHECK_HIP(hipMemcpyAsync(s_batch_key_layer_cache   + offset, s_batch[gid]->key_matmul   + b * kv_dim, kv_dim * sizeof(float), hipMemcpyDeviceToDevice, handle[gid].calc_stream));
//                 // CHECK_HIP(hipMemcpyAsync(s_batch_value_layer_cache + offset, s_batch[gid]->value_matmul + b * kv_dim, kv_dim * sizeof(float), hipMemcpyDeviceToDevice, handle[gid].calc_stream));
//                 kv_caches[gid]

//                 thablas_status = thaDNN_s_rope(&handle[gid], dim, head_size, kv_dim, pos[b], s_batch[gid]->q + b * dim, s_batch_key_layer_cache + offset);
//             }

//             // 3. Use kv_layer_cache to calculate multi-head attention
//             int multi_head_n_buffer_words = p->seq_len;
//             thablas_status = thaDNN_s_multiheads_1_v2_batch(&handle[gid], batch_size, pipe_size, pos, pos_d, p->n_heads, s_batch[gid]->q, s_batch[gid]->att, s_batch_key_layer_cache, head_size, multi_head_n_buffer_words, kv_dim, dim, kv_mul);
//             thablas_status = thaDNN_s_multiheads_2_batch(&handle[gid], batch_size, s_batch[gid]->att, pos_d, multi_head_n_buffer_words, p->n_heads);
//             thablas_status = thaDNN_s_multiheads_3_v2_batch(&handle[gid], batch_size, pos_d, p->n_heads, s_batch[gid]->xb, s_batch[gid]->att, s_batch_value_layer_cache, head_size, multi_head_n_buffer_words, kv_dim, kv_mul, dim, pipe_size);

//             thablas_status = thaBLAS_s_matmul_ifdef(&handle[gid], w[gid]->wo + l*dim*dim, s_batch[gid]->xb, s_batch[gid]->xb2, dim, batch_size, dim);

//             if (max_pos + 1 > n_buffer_words) 
//             {
//                 // 4. Copy kv_layer_cache back to host kv_cache
//                 int total_words_to_swap = max_pos + 1 - n_buffer_words;
//                 int total_buffer_words = n_buffer_words * batch_size * kv_dim;

//                 CHECK_HIP(hipMemcpyAsync(s_batch[gid]->key_cache + loff, s_batch_key_layer_cache, total_buffer_words * sizeof(float), hipMemcpyDeviceToDevice, handle[gid].copy_stream));
//                 CHECK_HIP(hipMemcpyAsync(s_batch[gid]->value_cache + loff, s_batch_value_layer_cache, total_buffer_words * sizeof(float), hipMemcpyDeviceToDevice, handle[gid].copy_stream));

//                 int host_loff = l * (p->seq_len - n_buffer_words) * batch_size * kv_dim;
//                 CHECK_HIP(hipMemcpyAsync(s_host_batch[gid]->key_cache + host_loff, s_batch_key_layer_cache + total_buffer_words, total_words_to_swap * batch_size * kv_dim * sizeof(float), hipMemcpyDeviceToHost, handle[gid].copy_stream));
//                 CHECK_HIP(hipMemcpyAsync(s_host_batch[gid]->value_cache + host_loff, s_batch_value_layer_cache + total_buffer_words, total_words_to_swap * batch_size * kv_dim * sizeof(float), hipMemcpyDeviceToHost, handle[gid].copy_stream));
//             }

//             for(int b=0 ; b<batch_size ; ++b)
//                 thablas_status = thaBLAS_s_vecaddvec(&handle[gid], s_batch[gid]->x + b * dim, s_batch[gid]->xb2 + b * dim, dim);

//             thablas_status = thaDNN_s_rmsnorm_v2_batch(&handle[gid], batch_size, s_batch[gid]->xb, s_batch[gid]->x, w[gid]->rms_ffn_weight + l*dim, dim, dim);

//             thablas_status = thaBLAS_s_matmul_ifdef(&handle[gid], w[gid]->w1 + l*dim*hidden_dim, s_batch[gid]->xb, s_batch[gid]->hb, hidden_dim, batch_size, dim);
//             thablas_status = thaBLAS_s_matmul_ifdef(&handle[gid], w[gid]->w3 + l*dim*hidden_dim, s_batch[gid]->xb, s_batch[gid]->hb2, hidden_dim, batch_size, dim);

//             for(int b=0 ; b<batch_size ; ++b)
//                 thablas_status = thaDNN_s_swiglu(&handle[gid], s_batch[gid]->hb + b * hidden_dim, s_batch[gid]->hb2 + b * hidden_dim, hidden_dim);

//             thablas_status = thaBLAS_s_matmul_ifdef(&handle[gid], w[gid]->w2 + l*dim*hidden_dim, s_batch[gid]->hb, s_batch[gid]->xb, dim, batch_size, hidden_dim);

//             for(int b=0 ; b<batch_size ; ++b)
//                 thablas_status = thaBLAS_s_vecaddvec(&handle[gid], s_batch[gid]->x + b * dim, s_batch[gid]->xb + b * dim, dim);
//             CHECK_HIP(hipDeviceSynchronize());
//         }

//         int next_device = gid + 1;
//         if (next_device < n_devices) {
//             CHECK_HIP(hipMemcpyAsync(s_batch[next_device]->x, s_batch[gid]->x, batch_size * dim * sizeof(float), hipMemcpyDeviceToDevice, handle[gid].calc_stream));
//         } else {
//             thablas_status = thaDNN_s_rmsnorm_v2_batch(&handle[gid], batch_size, s_batch[gid]->x, s_batch[gid]->x, w[gid]->rms_final_weight, dim, dim);
//             thablas_status = thaBLAS_s_matmul_ifdef(&handle[gid], w[gid]->wcls, s_batch[gid]->x, logits_host, p->vocab_size, batch_size, dim);
//         }
        
//         CHECK_HIP(hipDeviceSynchronize());
//         // for(unsigned long long l = 0; l < pipe_size; l++)
//         //     CHECK_HIP(hipEventDestroy(copy_done_event[l]));
//         CHECK_HIP(hipFree(pos_d));
//         omp_unset_lock(&device_locks[gid]);
//     }
     
//     return thablas_status;
// }


