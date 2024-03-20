#include "models.hpp"

// copy transformer checkpoint data from host to device
// all scalar values and pointer values are still stored on host
// only arrays are stored on device
void copy_transformer_to_device(thablasHandle_t handle, Transformer* t_h, Transformer* &t_d)
{
  Config *p = &t_h->config;
  int dim = p->dim;
  int vocab_size = p->vocab_size;
  int layer = p->n_layers;
  int n_layers = layer;
  int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
  int hidden_dim = p->hidden_dim;
  int n_heads = p->n_heads;
  int seq_len = p->seq_len;
  
  t_d = (Transformer*)malloc(sizeof(Transformer));
  // Config config; // the hyperparameters of the architecture (the blueprint)
  // TransformerWeights weights; // the weights of the model
  // RunState state; // buffers for the "wave" of activations in the forward pass
  // // some more state needed to properly clean up the memory mapping (sigh)
  // int fd; // file descriptor for memory mapping
  // float* data; // memory mapped data pointer
  // ssize_t file_size; // size of the checkpoint file in bytes
  memcpy(&t_d->config, p, sizeof(Config));

  CHECK_HIP(hipMalloc(&t_d->weights.token_embedding_table, vocab_size * dim * sizeof(float)));
  CHECK_HIP(hipMalloc(&t_d->weights.rms_att_weight, layer * dim * sizeof(float)));
  CHECK_HIP(hipMalloc(&t_d->weights.rms_ffn_weight, layer * dim * sizeof(float)));
  CHECK_HIP(hipMalloc(&t_d->weights.wq, layer * dim * dim * sizeof(float)));
  CHECK_HIP(hipMalloc(&t_d->weights.wk, layer * dim * kv_dim * sizeof(float)));
  CHECK_HIP(hipMalloc(&t_d->weights.wv, layer * dim * kv_dim * sizeof(float)));
  CHECK_HIP(hipMalloc(&t_d->weights.wo, layer * dim * dim * sizeof(float)));
  CHECK_HIP(hipMalloc(&t_d->weights.w1, layer * hidden_dim * dim * sizeof(float)));
  CHECK_HIP(hipMalloc(&t_d->weights.w2, layer * dim * hidden_dim * sizeof(float)));
  CHECK_HIP(hipMalloc(&t_d->weights.w3, layer * hidden_dim * dim * sizeof(float)));
  CHECK_HIP(hipMalloc(&t_d->weights.rms_final_weight, dim * sizeof(float)));
  CHECK_HIP(hipMalloc(&t_d->weights.wcls, dim * vocab_size * sizeof(float)));  
  
  CHECK_HIP(hipMemcpy(t_d->weights.token_embedding_table, t_h->weights.token_embedding_table, vocab_size * dim * sizeof(float), hipMemcpyHostToDevice));
  CHECK_HIP(hipMemcpy(t_d->weights.rms_att_weight, t_h->weights.rms_att_weight, layer * dim * sizeof(float), hipMemcpyHostToDevice));
  CHECK_HIP(hipMemcpy(t_d->weights.rms_ffn_weight, t_h->weights.rms_ffn_weight, layer * dim * sizeof(float), hipMemcpyHostToDevice));
  CHECK_HIP(hipMemcpy(t_d->weights.wq, t_h->weights.wq, layer * dim * dim * sizeof(float), hipMemcpyHostToDevice));
  CHECK_HIP(hipMemcpy(t_d->weights.wk, t_h->weights.wk, layer * dim * kv_dim * sizeof(float), hipMemcpyHostToDevice));
  CHECK_HIP(hipMemcpy(t_d->weights.wv, t_h->weights.wv, layer * dim * kv_dim * sizeof(float), hipMemcpyHostToDevice));
  CHECK_HIP(hipMemcpy(t_d->weights.wo, t_h->weights.wo, layer * dim * dim * sizeof(float), hipMemcpyHostToDevice));
  CHECK_HIP(hipMemcpy(t_d->weights.w1, t_h->weights.w1, layer * hidden_dim * dim * sizeof(float), hipMemcpyHostToDevice));
  CHECK_HIP(hipMemcpy(t_d->weights.w2, t_h->weights.w2, layer * dim * hidden_dim * sizeof(float), hipMemcpyHostToDevice));
  CHECK_HIP(hipMemcpy(t_d->weights.w3, t_h->weights.w3, layer * hidden_dim * dim * sizeof(float), hipMemcpyHostToDevice));
  CHECK_HIP(hipMemcpy(t_d->weights.rms_final_weight, t_h->weights.rms_final_weight, dim * sizeof(float), hipMemcpyHostToDevice));
  CHECK_HIP(hipMemcpy(t_d->weights.wcls, t_h->weights.wcls, dim * vocab_size * sizeof(float), hipMemcpyHostToDevice));

  CHECK_HIP(hipMalloc(&t_d->state.x, dim * sizeof(float)));
  CHECK_HIP(hipMalloc(&t_d->state.xb, dim * sizeof(float)));
  CHECK_HIP(hipMalloc(&t_d->state.xb2, dim * sizeof(float)));
  CHECK_HIP(hipMalloc(&t_d->state.hb, hidden_dim * sizeof(float)));
  CHECK_HIP(hipMalloc(&t_d->state.hb2, hidden_dim * sizeof(float)));
  CHECK_HIP(hipMalloc(&t_d->state.q, dim * sizeof(float)));
  // CHECK_HIP(hipMalloc(&t_d->state.k, dim * sizeof(float)));
  // CHECK_HIP(hipMalloc(&t_d->state.v, dim * sizeof(float)));
  CHECK_HIP(hipMalloc(&t_d->state.att, n_heads * seq_len * sizeof(float)));
  CHECK_HIP(hipMalloc(&t_d->state.logits, vocab_size * sizeof(float)));
  CHECK_HIP(hipMalloc(&t_d->state.key_cache, n_layers * seq_len * kv_dim * sizeof(float)));
  CHECK_HIP(hipMalloc(&t_d->state.value_cache, n_layers * seq_len * kv_dim * sizeof(float)));

  CHECK_HIP(hipMemcpy(t_d->state.x, t_h->state.x, dim * sizeof(float), hipMemcpyHostToDevice));
  CHECK_HIP(hipMemcpy(t_d->state.xb, t_h->state.xb, dim * sizeof(float), hipMemcpyHostToDevice));
  CHECK_HIP(hipMemcpy(t_d->state.xb2, t_h->state.xb2, dim * sizeof(float), hipMemcpyHostToDevice));
  CHECK_HIP(hipMemcpy(t_d->state.hb, t_h->state.hb, hidden_dim * sizeof(float), hipMemcpyHostToDevice));
  CHECK_HIP(hipMemcpy(t_d->state.hb2, t_h->state.hb2, hidden_dim * sizeof(float), hipMemcpyHostToDevice));
  CHECK_HIP(hipMemcpy(t_d->state.q, t_h->state.q, dim * sizeof(float), hipMemcpyHostToDevice));
  // CHECK_HIP(hipMemcpy(t_d->state.k, t_h->state.k, dim * sizeof(float), hipMemcpyHostToDevice));
  // CHECK_HIP(hipMemcpy(t_d->state.v, t_h->state.v, dim * sizeof(float), hipMemcpyHostToDevice));
  CHECK_HIP(hipMemcpy(t_d->state.att, t_h->state.att, n_heads * seq_len * sizeof(float), hipMemcpyHostToDevice));
  CHECK_HIP(hipMemcpy(t_d->state.logits, t_h->state.logits, vocab_size * sizeof(float), hipMemcpyHostToDevice));
  CHECK_HIP(hipMemcpy(t_d->state.key_cache, t_h->state.key_cache, n_layers * seq_len * kv_dim * sizeof(float), hipMemcpyHostToDevice));
  CHECK_HIP(hipMemcpy(t_d->state.value_cache, t_h->state.value_cache, n_layers * seq_len * kv_dim * sizeof(float), hipMemcpyHostToDevice));
}

// all pointer values are still stored on host
// only arrays are stored on device
void copy_weight_to_device(Transformer* t_h, TransformerWeights* &w_d)
{
  Config *p = &t_h->config;
  int dim = p->dim;
  int vocab_size = p->vocab_size;
  int layer = p->n_layers;
  // int n_layers = layer;
  int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
  int hidden_dim = p->hidden_dim;
  // int n_heads = p->n_heads;
  // int seq_len = p->seq_len;

  w_d = (TransformerWeights*)malloc(sizeof(TransformerWeights));

  CHECK_HIP(hipMalloc(&w_d->token_embedding_table, vocab_size * dim * sizeof(float)));
  CHECK_HIP(hipMalloc(&w_d->rms_att_weight, layer * dim * sizeof(float)));
  CHECK_HIP(hipMalloc(&w_d->rms_ffn_weight, layer * dim * sizeof(float)));
  CHECK_HIP(hipMalloc(&w_d->wq, layer * dim * dim * sizeof(float)));
  CHECK_HIP(hipMalloc(&w_d->wk, layer * dim * kv_dim * sizeof(float)));
  CHECK_HIP(hipMalloc(&w_d->wv, layer * dim * kv_dim * sizeof(float)));
  CHECK_HIP(hipMalloc(&w_d->wo, layer * dim * dim * sizeof(float)));
  CHECK_HIP(hipMalloc(&w_d->w1, layer * hidden_dim * dim * sizeof(float)));
  CHECK_HIP(hipMalloc(&w_d->w2, layer * dim * hidden_dim * sizeof(float)));
  CHECK_HIP(hipMalloc(&w_d->w3, layer * hidden_dim * dim * sizeof(float)));
  CHECK_HIP(hipMalloc(&w_d->rms_final_weight, dim * sizeof(float)));
  CHECK_HIP(hipMalloc(&w_d->wcls, dim * vocab_size * sizeof(float)));  
  
  CHECK_HIP(hipMemcpy(w_d->token_embedding_table, t_h->weights.token_embedding_table, vocab_size * dim * sizeof(float), hipMemcpyHostToDevice));
  CHECK_HIP(hipMemcpy(w_d->rms_att_weight, t_h->weights.rms_att_weight, layer * dim * sizeof(float), hipMemcpyHostToDevice));
  CHECK_HIP(hipMemcpy(w_d->rms_ffn_weight, t_h->weights.rms_ffn_weight, layer * dim * sizeof(float), hipMemcpyHostToDevice));
  CHECK_HIP(hipMemcpy(w_d->wq, t_h->weights.wq, layer * dim * dim * sizeof(float), hipMemcpyHostToDevice));
  CHECK_HIP(hipMemcpy(w_d->wk, t_h->weights.wk, layer * dim * kv_dim * sizeof(float), hipMemcpyHostToDevice));
  CHECK_HIP(hipMemcpy(w_d->wv, t_h->weights.wv, layer * dim * kv_dim * sizeof(float), hipMemcpyHostToDevice));
  CHECK_HIP(hipMemcpy(w_d->wo, t_h->weights.wo, layer * dim * dim * sizeof(float), hipMemcpyHostToDevice));
  CHECK_HIP(hipMemcpy(w_d->w1, t_h->weights.w1, layer * hidden_dim * dim * sizeof(float), hipMemcpyHostToDevice));
  CHECK_HIP(hipMemcpy(w_d->w2, t_h->weights.w2, layer * dim * hidden_dim * sizeof(float), hipMemcpyHostToDevice));
  CHECK_HIP(hipMemcpy(w_d->w3, t_h->weights.w3, layer * hidden_dim * dim * sizeof(float), hipMemcpyHostToDevice));
  CHECK_HIP(hipMemcpy(w_d->rms_final_weight, t_h->weights.rms_final_weight, dim * sizeof(float), hipMemcpyHostToDevice));
  CHECK_HIP(hipMemcpy(w_d->wcls, t_h->weights.wcls, dim * vocab_size * sizeof(float), hipMemcpyHostToDevice));
}

// all pointer values are still stored on host
// only arrays are stored on device
void alloc_state_to_device(Transformer* t_h, RunState* &s_d)
{
  Config *p = &t_h->config;
  int dim = p->dim;
  int vocab_size = p->vocab_size;
  int layer = p->n_layers;
  int n_layers = layer;
  int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
  int hidden_dim = p->hidden_dim;
  int n_heads = p->n_heads;
  int seq_len = p->seq_len;

  s_d = (RunState*)malloc(sizeof(RunState));

  CHECK_HIP(hipMalloc(&s_d->x, dim * sizeof(float)));
  CHECK_HIP(hipMalloc(&s_d->xb, dim * sizeof(float)));
  CHECK_HIP(hipMalloc(&s_d->xb2, dim * sizeof(float)));
  CHECK_HIP(hipMalloc(&s_d->hb, hidden_dim * sizeof(float)));
  CHECK_HIP(hipMalloc(&s_d->hb2, hidden_dim * sizeof(float)));
  CHECK_HIP(hipMalloc(&s_d->q, dim * sizeof(float)));
  CHECK_HIP(hipMalloc(&s_d->att, n_heads * seq_len * sizeof(float)));
  CHECK_HIP(hipMalloc(&s_d->logits, vocab_size * sizeof(float)));
  CHECK_HIP(hipMalloc(&s_d->key_cache, n_layers * seq_len * kv_dim * sizeof(float)));
  CHECK_HIP(hipMalloc(&s_d->value_cache, n_layers * seq_len * kv_dim * sizeof(float)));
}

void alloc_state_to_device_batch(Transformer* t_h, RunState* &s_d_batch, int batch_size)
{
  Config *p = &t_h->config;
  int dim = p->dim;
  int vocab_size = p->vocab_size;
  int layer = p->n_layers;
  int n_layers = layer;
  int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
  int hidden_dim = p->hidden_dim;
  int n_heads = p->n_heads;
  int seq_len = p->seq_len;

  s_d_batch = (RunState*)malloc(sizeof(RunState));

  CHECK_HIP(hipMalloc(&s_d_batch->x, dim * sizeof(float) * batch_size));
  CHECK_HIP(hipMalloc(&s_d_batch->xb, dim * sizeof(float) * batch_size));
  CHECK_HIP(hipMalloc(&s_d_batch->xb2, dim * sizeof(float) * batch_size));
  CHECK_HIP(hipMalloc(&s_d_batch->hb, hidden_dim * sizeof(float) * batch_size));
  CHECK_HIP(hipMalloc(&s_d_batch->hb2, hidden_dim * sizeof(float) * batch_size));
  CHECK_HIP(hipMalloc(&s_d_batch->q, dim * sizeof(float) * batch_size));
  CHECK_HIP(hipMalloc(&s_d_batch->att, n_heads * seq_len * sizeof(float) * batch_size));
  CHECK_HIP(hipMalloc(&s_d_batch->logits, vocab_size * sizeof(float) * batch_size));
  CHECK_HIP(hipMalloc(&s_d_batch->key_cache, n_layers * seq_len * kv_dim * sizeof(float) * batch_size));
  CHECK_HIP(hipMalloc(&s_d_batch->value_cache, n_layers * seq_len * kv_dim * sizeof(float) * batch_size));
}

void copy_transformer_pipeline_to_device(thablasHandle_t handle, Transformer* t_h, Transformer* &t_d, int pipe_size, int pipe_id)
{
  Config *p = &t_h->config;
  int dim = p->dim;
  int vocab_size = p->vocab_size;
  int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
  int hidden_dim = p->hidden_dim;
  int n_heads = p->n_heads;
  int seq_len = p->seq_len;

  t_d = (Transformer*)malloc(sizeof(Transformer));
  // Config config; // the hyperparameters of the architecture (the blueprint)
  // TransformerWeights weights; // the weights of the model
  // RunState state; // buffers for the "wave" of activations in the forward pass
  // // some more state needed to properly clean up the memory mapping (sigh)
  // int fd; // file descriptor for memory mapping
  // float* data; // memory mapped data pointer
  // ssize_t file_size; // size of the checkpoint file in bytes
  memcpy(&t_d->config, p, sizeof(Config));

  CHECK_HIP(hipMalloc(&t_d->weights.token_embedding_table, vocab_size * dim * sizeof(float)));
  CHECK_HIP(hipMalloc(&t_d->weights.rms_att_weight, pipe_size * dim * sizeof(float)));
  CHECK_HIP(hipMalloc(&t_d->weights.rms_ffn_weight, pipe_size * dim * sizeof(float)));
  CHECK_HIP(hipMalloc(&t_d->weights.wq, pipe_size * dim * dim * sizeof(float)));
  CHECK_HIP(hipMalloc(&t_d->weights.wk, pipe_size * dim * kv_dim * sizeof(float)));
  CHECK_HIP(hipMalloc(&t_d->weights.wv, pipe_size * dim * kv_dim * sizeof(float)));
  CHECK_HIP(hipMalloc(&t_d->weights.wo, pipe_size * dim * dim * sizeof(float)));
  CHECK_HIP(hipMalloc(&t_d->weights.w1, pipe_size * hidden_dim * dim * sizeof(float)));
  CHECK_HIP(hipMalloc(&t_d->weights.w2, pipe_size * dim * hidden_dim * sizeof(float)));
  CHECK_HIP(hipMalloc(&t_d->weights.w3, pipe_size * hidden_dim * dim * sizeof(float)));
  CHECK_HIP(hipMalloc(&t_d->weights.rms_final_weight, dim * sizeof(float)));
  CHECK_HIP(hipMalloc(&t_d->weights.wcls, dim * vocab_size * sizeof(float)));  

  int poff = pipe_id * pipe_size;
  CHECK_HIP(hipMemcpy(t_d->weights.token_embedding_table, t_h->weights.token_embedding_table, vocab_size * dim * sizeof(float), hipMemcpyHostToDevice));
  CHECK_HIP(hipMemcpy(t_d->weights.rms_att_weight,        t_h->weights.rms_att_weight + poff * dim,  pipe_size * dim * sizeof(float), hipMemcpyHostToDevice));
  CHECK_HIP(hipMemcpy(t_d->weights.rms_ffn_weight,        t_h->weights.rms_ffn_weight + poff * dim,  pipe_size * dim * sizeof(float), hipMemcpyHostToDevice));
  CHECK_HIP(hipMemcpy(t_d->weights.wq,                    t_h->weights.wq + poff * dim * dim,        pipe_size * dim * dim * sizeof(float), hipMemcpyHostToDevice));
  CHECK_HIP(hipMemcpy(t_d->weights.wk,                    t_h->weights.wk + poff * dim * kv_dim,     pipe_size * dim * kv_dim * sizeof(float), hipMemcpyHostToDevice));
  CHECK_HIP(hipMemcpy(t_d->weights.wv,                    t_h->weights.wv + poff * dim * kv_dim,     pipe_size * dim * kv_dim * sizeof(float), hipMemcpyHostToDevice));
  CHECK_HIP(hipMemcpy(t_d->weights.wo,                    t_h->weights.wo + poff * dim * dim,        pipe_size * dim * dim * sizeof(float), hipMemcpyHostToDevice));
  CHECK_HIP(hipMemcpy(t_d->weights.w1,                    t_h->weights.w1 + poff * hidden_dim * dim, pipe_size * hidden_dim * dim * sizeof(float), hipMemcpyHostToDevice));
  CHECK_HIP(hipMemcpy(t_d->weights.w2,                    t_h->weights.w2 + poff * dim * hidden_dim, pipe_size * dim * hidden_dim * sizeof(float), hipMemcpyHostToDevice));
  CHECK_HIP(hipMemcpy(t_d->weights.w3,                    t_h->weights.w3 + poff * hidden_dim * dim, pipe_size * hidden_dim * dim * sizeof(float), hipMemcpyHostToDevice));
  CHECK_HIP(hipMemcpy(t_d->weights.rms_final_weight, t_h->weights.rms_final_weight, dim * sizeof(float), hipMemcpyHostToDevice));
  CHECK_HIP(hipMemcpy(t_d->weights.wcls, t_h->weights.wcls, dim * vocab_size * sizeof(float), hipMemcpyHostToDevice));

  CHECK_HIP(hipMalloc(&t_d->state.x, dim * sizeof(float)));
  CHECK_HIP(hipMalloc(&t_d->state.xb, dim * sizeof(float)));
  CHECK_HIP(hipMalloc(&t_d->state.xb2, dim * sizeof(float)));
  CHECK_HIP(hipMalloc(&t_d->state.hb, hidden_dim * sizeof(float)));
  CHECK_HIP(hipMalloc(&t_d->state.hb2, hidden_dim * sizeof(float)));
  CHECK_HIP(hipMalloc(&t_d->state.q, dim * sizeof(float)));
  // CHECK_HIP(hipMalloc(&t_d->state.k, dim * sizeof(float)));
  // CHECK_HIP(hipMalloc(&t_d->state.v, dim * sizeof(float)));
  CHECK_HIP(hipMalloc(&t_d->state.att, n_heads * seq_len * sizeof(float)));
  CHECK_HIP(hipMalloc(&t_d->state.logits, vocab_size * sizeof(float)));
  CHECK_HIP(hipMalloc(&t_d->state.key_cache, pipe_size * seq_len * kv_dim * sizeof(float)));
  CHECK_HIP(hipMalloc(&t_d->state.value_cache, pipe_size * seq_len * kv_dim * sizeof(float)));

  // CHECK_HIP(hipMemcpy(t_d->state.x, t_h->state.x, dim * sizeof(float), hipMemcpyHostToDevice));
  // CHECK_HIP(hipMemcpy(t_d->state.xb, t_h->state.xb, dim * sizeof(float), hipMemcpyHostToDevice));
  // CHECK_HIP(hipMemcpy(t_d->state.xb2, t_h->state.xb2, dim * sizeof(float), hipMemcpyHostToDevice));
  // CHECK_HIP(hipMemcpy(t_d->state.hb, t_h->state.hb, hidden_dim * sizeof(float), hipMemcpyHostToDevice));
  // CHECK_HIP(hipMemcpy(t_d->state.hb2, t_h->state.hb2, hidden_dim * sizeof(float), hipMemcpyHostToDevice));
  // CHECK_HIP(hipMemcpy(t_d->state.q, t_h->state.q, dim * sizeof(float), hipMemcpyHostToDevice));
  // // CHECK_HIP(hipMemcpy(t_d->state.k, t_h->state.k, dim * sizeof(float), hipMemcpyHostToDevice));
  // // CHECK_HIP(hipMemcpy(t_d->state.v, t_h->state.v, dim * sizeof(float), hipMemcpyHostToDevice));
  // CHECK_HIP(hipMemcpy(t_d->state.att, t_h->state.att, n_heads * seq_len * sizeof(float), hipMemcpyHostToDevice));
  // CHECK_HIP(hipMemcpy(t_d->state.logits, t_h->state.logits, vocab_size * sizeof(float), hipMemcpyHostToDevice));
  // CHECK_HIP(hipMemcpy(t_d->state.key_cache, t_h->state.key_cache, pipe_size * seq_len * kv_dim * sizeof(float), hipMemcpyHostToDevice));
  // CHECK_HIP(hipMemcpy(t_d->state.value_cache, t_h->state.value_cache, pipe_size * seq_len * kv_dim * sizeof(float), hipMemcpyHostToDevice));
}

void copy_transformer_pipeline_to_device_batch(thablasHandle_t handle, Transformer* t_h, Transformer* &t_d, int pipe_size, int pipe_id, int batch_size)
{
  Config *p = &t_h->config;
  int dim = p->dim;
  int vocab_size = p->vocab_size;
  int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
  int hidden_dim = p->hidden_dim;
  int n_heads = p->n_heads;
  int seq_len = p->seq_len;

  t_d = (Transformer*)malloc(sizeof(Transformer));
  // Config config; // the hyperparameters of the architecture (the blueprint)
  // TransformerWeights weights; // the weights of the model
  // RunState state; // buffers for the "wave" of activations in the forward pass
  // // some more state needed to properly clean up the memory mapping (sigh)
  // int fd; // file descriptor for memory mapping
  // float* data; // memory mapped data pointer
  // ssize_t file_size; // size of the checkpoint file in bytes
  memcpy(&t_d->config, p, sizeof(Config));

  CHECK_HIP(hipMalloc(&t_d->weights.token_embedding_table, vocab_size * dim * sizeof(float)));
  CHECK_HIP(hipMalloc(&t_d->weights.rms_att_weight, pipe_size * dim * sizeof(float)));
  CHECK_HIP(hipMalloc(&t_d->weights.rms_ffn_weight, pipe_size * dim * sizeof(float)));
  CHECK_HIP(hipMalloc(&t_d->weights.wq, pipe_size * dim * dim * sizeof(float)));
  CHECK_HIP(hipMalloc(&t_d->weights.wk, pipe_size * dim * kv_dim * sizeof(float)));
  CHECK_HIP(hipMalloc(&t_d->weights.wv, pipe_size * dim * kv_dim * sizeof(float)));
  CHECK_HIP(hipMalloc(&t_d->weights.wo, pipe_size * dim * dim * sizeof(float)));
  CHECK_HIP(hipMalloc(&t_d->weights.w1, pipe_size * hidden_dim * dim * sizeof(float)));
  CHECK_HIP(hipMalloc(&t_d->weights.w2, pipe_size * dim * hidden_dim * sizeof(float)));
  CHECK_HIP(hipMalloc(&t_d->weights.w3, pipe_size * hidden_dim * dim * sizeof(float)));
  CHECK_HIP(hipMalloc(&t_d->weights.rms_final_weight, dim * sizeof(float)));
  CHECK_HIP(hipMalloc(&t_d->weights.wcls, dim * vocab_size * sizeof(float)));  

  int poff = pipe_id * pipe_size;
  CHECK_HIP(hipMemcpy(t_d->weights.token_embedding_table, t_h->weights.token_embedding_table, vocab_size * dim * sizeof(float), hipMemcpyHostToDevice));
  CHECK_HIP(hipMemcpy(t_d->weights.rms_att_weight,        t_h->weights.rms_att_weight + poff * dim,  pipe_size * dim * sizeof(float), hipMemcpyHostToDevice));
  CHECK_HIP(hipMemcpy(t_d->weights.rms_ffn_weight,        t_h->weights.rms_ffn_weight + poff * dim,  pipe_size * dim * sizeof(float), hipMemcpyHostToDevice));
  CHECK_HIP(hipMemcpy(t_d->weights.wq,                    t_h->weights.wq + poff * dim * dim,        pipe_size * dim * dim * sizeof(float), hipMemcpyHostToDevice));
  CHECK_HIP(hipMemcpy(t_d->weights.wk,                    t_h->weights.wk + poff * dim * kv_dim,     pipe_size * dim * kv_dim * sizeof(float), hipMemcpyHostToDevice));
  CHECK_HIP(hipMemcpy(t_d->weights.wv,                    t_h->weights.wv + poff * dim * kv_dim,     pipe_size * dim * kv_dim * sizeof(float), hipMemcpyHostToDevice));
  CHECK_HIP(hipMemcpy(t_d->weights.wo,                    t_h->weights.wo + poff * dim * dim,        pipe_size * dim * dim * sizeof(float), hipMemcpyHostToDevice));
  CHECK_HIP(hipMemcpy(t_d->weights.w1,                    t_h->weights.w1 + poff * hidden_dim * dim, pipe_size * hidden_dim * dim * sizeof(float), hipMemcpyHostToDevice));
  CHECK_HIP(hipMemcpy(t_d->weights.w2,                    t_h->weights.w2 + poff * dim * hidden_dim, pipe_size * dim * hidden_dim * sizeof(float), hipMemcpyHostToDevice));
  CHECK_HIP(hipMemcpy(t_d->weights.w3,                    t_h->weights.w3 + poff * hidden_dim * dim, pipe_size * hidden_dim * dim * sizeof(float), hipMemcpyHostToDevice));
  CHECK_HIP(hipMemcpy(t_d->weights.rms_final_weight, t_h->weights.rms_final_weight, dim * sizeof(float), hipMemcpyHostToDevice));
  CHECK_HIP(hipMemcpy(t_d->weights.wcls, t_h->weights.wcls, dim * vocab_size * sizeof(float), hipMemcpyHostToDevice));

  CHECK_HIP(hipMalloc(&t_d->state.x, dim * sizeof(float) * batch_size));
  CHECK_HIP(hipMalloc(&t_d->state.xb, dim * sizeof(float) * batch_size));
  CHECK_HIP(hipMalloc(&t_d->state.xb2, dim * sizeof(float) * batch_size));
  CHECK_HIP(hipMalloc(&t_d->state.hb, hidden_dim * sizeof(float) * batch_size));
  CHECK_HIP(hipMalloc(&t_d->state.hb2, hidden_dim * sizeof(float) * batch_size));
  CHECK_HIP(hipMalloc(&t_d->state.q, dim * sizeof(float) * batch_size));
  CHECK_HIP(hipMalloc(&t_d->state.att, n_heads * seq_len * sizeof(float) * batch_size));
  CHECK_HIP(hipMalloc(&t_d->state.logits, vocab_size * sizeof(float) * batch_size));
  CHECK_HIP(hipMalloc(&t_d->state.key_cache, pipe_size * seq_len * kv_dim * sizeof(float) * batch_size));
  CHECK_HIP(hipMalloc(&t_d->state.value_cache, pipe_size * seq_len * kv_dim * sizeof(float) * batch_size));

  // CHECK_HIP(hipMemcpy(t_d->state.x, t_h->state.x, dim * sizeof(float), hipMemcpyHostToDevice));
  // CHECK_HIP(hipMemcpy(t_d->state.xb, t_h->state.xb, dim * sizeof(float), hipMemcpyHostToDevice));
  // CHECK_HIP(hipMemcpy(t_d->state.xb2, t_h->state.xb2, dim * sizeof(float), hipMemcpyHostToDevice));
  // CHECK_HIP(hipMemcpy(t_d->state.hb, t_h->state.hb, hidden_dim * sizeof(float), hipMemcpyHostToDevice));
  // CHECK_HIP(hipMemcpy(t_d->state.hb2, t_h->state.hb2, hidden_dim * sizeof(float), hipMemcpyHostToDevice));
  // CHECK_HIP(hipMemcpy(t_d->state.q, t_h->state.q, dim * sizeof(float), hipMemcpyHostToDevice));
  // // CHECK_HIP(hipMemcpy(t_d->state.k, t_h->state.k, dim * sizeof(float), hipMemcpyHostToDevice));
  // // CHECK_HIP(hipMemcpy(t_d->state.v, t_h->state.v, dim * sizeof(float), hipMemcpyHostToDevice));
  // CHECK_HIP(hipMemcpy(t_d->state.att, t_h->state.att, n_heads * seq_len * sizeof(float), hipMemcpyHostToDevice));
  // CHECK_HIP(hipMemcpy(t_d->state.logits, t_h->state.logits, vocab_size * sizeof(float), hipMemcpyHostToDevice));
  // CHECK_HIP(hipMemcpy(t_d->state.key_cache, t_h->state.key_cache, pipe_size * seq_len * kv_dim * sizeof(float), hipMemcpyHostToDevice));
  // CHECK_HIP(hipMemcpy(t_d->state.value_cache, t_h->state.value_cache, pipe_size * seq_len * kv_dim * sizeof(float), hipMemcpyHostToDevice));
}

void copy_transformer_weight_pipeline_to_device_batch(thablasHandle_t handle, Transformer* t_h, TransformerWeights* &w_d, int pipe_size, int pipe_id, int batch_size)
{
  Config *p = &t_h->config;
  int dim = p->dim;
  int vocab_size = p->vocab_size;
  int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
  int hidden_dim = p->hidden_dim;
  // int n_heads = p->n_heads;
  // int seq_len = p->seq_len;

  w_d = (TransformerWeights*)malloc(sizeof(TransformerWeights));
  // Config config; // the hyperparameters of the architecture (the blueprint)
  // TransformerWeights weights; // the weights of the model
  // RunState state; // buffers for the "wave" of activations in the forward pass
  // // some more state needed to properly clean up the memory mapping (sigh)
  // int fd; // file descriptor for memory mapping
  // float* data; // memory mapped data pointer
  // ssize_t file_size; // size of the checkpoint file in bytes

  CHECK_HIP(hipMalloc(&w_d->token_embedding_table, vocab_size * dim * sizeof(float)));
  CHECK_HIP(hipMalloc(&w_d->rms_att_weight, pipe_size * dim * sizeof(float)));
  CHECK_HIP(hipMalloc(&w_d->rms_ffn_weight, pipe_size * dim * sizeof(float)));
  CHECK_HIP(hipMalloc(&w_d->wq, pipe_size * dim * dim * sizeof(float)));
  CHECK_HIP(hipMalloc(&w_d->wk, pipe_size * dim * kv_dim * sizeof(float)));
  CHECK_HIP(hipMalloc(&w_d->wv, pipe_size * dim * kv_dim * sizeof(float)));
  CHECK_HIP(hipMalloc(&w_d->wo, pipe_size * dim * dim * sizeof(float)));
  CHECK_HIP(hipMalloc(&w_d->w1, pipe_size * hidden_dim * dim * sizeof(float)));
  CHECK_HIP(hipMalloc(&w_d->w2, pipe_size * dim * hidden_dim * sizeof(float)));
  CHECK_HIP(hipMalloc(&w_d->w3, pipe_size * hidden_dim * dim * sizeof(float)));
  CHECK_HIP(hipMalloc(&w_d->rms_final_weight, dim * sizeof(float)));
  CHECK_HIP(hipMalloc(&w_d->wcls, dim * vocab_size * sizeof(float)));  

  int poff = pipe_id * pipe_size;
  CHECK_HIP(hipMemcpy(w_d->token_embedding_table, t_h->weights.token_embedding_table, vocab_size * dim * sizeof(float), hipMemcpyHostToDevice));
  CHECK_HIP(hipMemcpy(w_d->rms_att_weight,        t_h->weights.rms_att_weight + poff * dim,  pipe_size * dim * sizeof(float), hipMemcpyHostToDevice));
  CHECK_HIP(hipMemcpy(w_d->rms_ffn_weight,        t_h->weights.rms_ffn_weight + poff * dim,  pipe_size * dim * sizeof(float), hipMemcpyHostToDevice));
  CHECK_HIP(hipMemcpy(w_d->wq,                    t_h->weights.wq + poff * dim * dim,        pipe_size * dim * dim * sizeof(float), hipMemcpyHostToDevice));
  CHECK_HIP(hipMemcpy(w_d->wk,                    t_h->weights.wk + poff * dim * kv_dim,     pipe_size * dim * kv_dim * sizeof(float), hipMemcpyHostToDevice));
  CHECK_HIP(hipMemcpy(w_d->wv,                    t_h->weights.wv + poff * dim * kv_dim,     pipe_size * dim * kv_dim * sizeof(float), hipMemcpyHostToDevice));
  CHECK_HIP(hipMemcpy(w_d->wo,                    t_h->weights.wo + poff * dim * dim,        pipe_size * dim * dim * sizeof(float), hipMemcpyHostToDevice));
  CHECK_HIP(hipMemcpy(w_d->w1,                    t_h->weights.w1 + poff * hidden_dim * dim, pipe_size * hidden_dim * dim * sizeof(float), hipMemcpyHostToDevice));
  CHECK_HIP(hipMemcpy(w_d->w2,                    t_h->weights.w2 + poff * dim * hidden_dim, pipe_size * dim * hidden_dim * sizeof(float), hipMemcpyHostToDevice));
  CHECK_HIP(hipMemcpy(w_d->w3,                    t_h->weights.w3 + poff * hidden_dim * dim, pipe_size * hidden_dim * dim * sizeof(float), hipMemcpyHostToDevice));
  CHECK_HIP(hipMemcpy(w_d->rms_final_weight, t_h->weights.rms_final_weight, dim * sizeof(float), hipMemcpyHostToDevice));
  CHECK_HIP(hipMemcpy(w_d->wcls, t_h->weights.wcls, dim * vocab_size * sizeof(float), hipMemcpyHostToDevice));
}

void alloc_run_state_to_device_batch(thablasHandle_t handle, Transformer* t_h, RunState* &s_d, int pipe_size, int pipe_id, int batch_size)
{
  Config *p = &t_h->config;
  int dim = p->dim;
  int vocab_size = p->vocab_size;
  int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
  int hidden_dim = p->hidden_dim;
  int n_heads = p->n_heads;
  int seq_len = p->seq_len;

  s_d = (RunState*)malloc(sizeof(RunState));
  // Config config; // the hyperparameters of the architecture (the blueprint)
  // TransformerWeights weights; // the weights of the model
  // RunState state; // buffers for the "wave" of activations in the forward pass
  // // some more state needed to properly clean up the memory mapping (sigh)
  // int fd; // file descriptor for memory mapping
  // float* data; // memory mapped data pointer
  // ssize_t file_size; // size of the checkpoint file in bytes

  CHECK_HIP(hipMalloc(&s_d->x, dim * sizeof(float) * batch_size));
  CHECK_HIP(hipMalloc(&s_d->xb, dim * sizeof(float) * batch_size));
  CHECK_HIP(hipMalloc(&s_d->xb2, dim * sizeof(float) * batch_size));
  CHECK_HIP(hipMalloc(&s_d->hb, hidden_dim * sizeof(float) * batch_size));
  CHECK_HIP(hipMalloc(&s_d->hb2, hidden_dim * sizeof(float) * batch_size));
  CHECK_HIP(hipMalloc(&s_d->q, dim * sizeof(float) * batch_size));
  CHECK_HIP(hipMalloc(&s_d->att, n_heads * seq_len * sizeof(float) * batch_size));
  CHECK_HIP(hipMalloc(&s_d->logits, vocab_size * sizeof(float) * batch_size));
  CHECK_HIP(hipMalloc(&s_d->key_cache, pipe_size * seq_len * kv_dim * sizeof(float) * batch_size));
  CHECK_HIP(hipMalloc(&s_d->value_cache, pipe_size * seq_len * kv_dim * sizeof(float) * batch_size));
  CHECK_HIP(hipMalloc(&s_d->key_matmul, kv_dim * sizeof(float) * batch_size));
  CHECK_HIP(hipMalloc(&s_d->value_matmul, kv_dim * sizeof(float) * batch_size));
  CHECK_HIP(hipMalloc(&s_d->key_matmul, kv_dim * sizeof(float) * batch_size));
  CHECK_HIP(hipMalloc(&s_d->value_matmul, kv_dim * sizeof(float) * batch_size));
}

// HOST ALLOCATION
void alloc_swap_run_state_on_host_batch(thablasHandle_t handle, Transformer* t_h, RunState* &s_h, int pipe_size, int pipe_id, int batch_size, int n_buffer_words) {
  Config *p = &t_h->config;
  int dim = p->dim;
  int vocab_size = p->vocab_size;
  int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
  int hidden_dim = p->hidden_dim;
  int n_heads = p->n_heads;
  // int seq_len = p->seq_len;
  int n_swap_words = p->seq_len - n_buffer_words; // + 2 just in case

  s_h = (RunState*)malloc(sizeof(RunState));
  // Config config; // the hyperparameters of the architecture (the blueprint)
  // TransformerWeights weights; // the weights of the model
  // RunState state; // buffers for the "wave" of activations in the forward pass
  // // some more state needed to properly clean up the memory mapping (sigh)
  // int fd; // file descriptor for memory mapping
  // float* data; // memory mapped data pointer
  // ssize_t file_size; // size of the checkpoint file in bytes

  CHECK_HIP(hipHostMalloc(&s_h->x, dim * sizeof(float) * batch_size));
  CHECK_HIP(hipHostMalloc(&s_h->xb, dim * sizeof(float) * batch_size));
  CHECK_HIP(hipHostMalloc(&s_h->xb2, dim * sizeof(float) * batch_size));
  CHECK_HIP(hipHostMalloc(&s_h->hb, hidden_dim * sizeof(float) * batch_size));
  CHECK_HIP(hipHostMalloc(&s_h->hb2, hidden_dim * sizeof(float) * batch_size));
  CHECK_HIP(hipHostMalloc(&s_h->q, dim * sizeof(float) * batch_size));
  CHECK_HIP(hipHostMalloc(&s_h->att, n_heads * n_swap_words * sizeof(float) * batch_size));
  CHECK_HIP(hipHostMalloc(&s_h->logits, vocab_size * sizeof(float) * batch_size));
  CHECK_HIP(hipHostMalloc(&s_h->key_cache, pipe_size * n_swap_words * kv_dim * sizeof(float) * batch_size));
  CHECK_HIP(hipHostMalloc(&s_h->value_cache, pipe_size * n_swap_words * kv_dim * sizeof(float) * batch_size));
}

// DEVICE ALLOCATION
void alloc_swap_run_state_to_device_batch(thablasHandle_t handle, Transformer* t_h, RunState* &s_d, int pipe_size, int pipe_id, int batch_size, int n_buffer_words) {
  Config *p = &t_h->config;
  int dim = p->dim;
  int vocab_size = p->vocab_size;
  int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
  int hidden_dim = p->hidden_dim;
  int n_heads = p->n_heads;
  // int seq_len = p->seq_len;

  s_d = (RunState*)malloc(sizeof(RunState));
  // Config config; // the hyperparameters of the architecture (the blueprint)
  // TransformerWeights weights; // the weights of the model
  // RunState state; // buffers for the "wave" of activations in the forward pass
  // // some more state needed to properly clean up the memory mapping (sigh)
  // int fd; // file descriptor for memory mapping
  // float* data; // memory mapped data pointer
  // ssize_t file_size; // size of the checkpoint file in bytes

  CHECK_HIP(hipMalloc(&s_d->x, dim * sizeof(float) * batch_size));
  CHECK_HIP(hipMalloc(&s_d->xb, dim * sizeof(float) * batch_size));
  CHECK_HIP(hipMalloc(&s_d->xb2, dim * sizeof(float) * batch_size));
  CHECK_HIP(hipMalloc(&s_d->hb, hidden_dim * sizeof(float) * batch_size));
  CHECK_HIP(hipMalloc(&s_d->hb2, hidden_dim * sizeof(float) * batch_size));
  CHECK_HIP(hipMalloc(&s_d->q, dim * sizeof(float) * batch_size));
  CHECK_HIP(hipMalloc(&s_d->att, n_heads * p->seq_len * sizeof(float) * batch_size));
  CHECK_HIP(hipMalloc(&s_d->logits, vocab_size * sizeof(float) * batch_size));

  CHECK_HIP(hipMalloc(&s_d->key_cache, pipe_size * n_buffer_words * kv_dim * sizeof(float) * batch_size));
  CHECK_HIP(hipMalloc(&s_d->value_cache, pipe_size * n_buffer_words * kv_dim * sizeof(float) * batch_size));

  CHECK_HIP(hipMalloc(&s_d->key_matmul,   kv_dim * sizeof(float) * batch_size));
  CHECK_HIP(hipMalloc(&s_d->value_matmul, kv_dim * sizeof(float) * batch_size));
  CHECK_HIP(hipMalloc(&s_d->key_layer_cache,   p->seq_len * kv_dim * sizeof(float) * batch_size));
  CHECK_HIP(hipMalloc(&s_d->value_layer_cache, p->seq_len * kv_dim * sizeof(float) * batch_size));
}

void free_transformer_device() {}
