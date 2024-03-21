#pragma once
#include <stdlib.h>
#include <ctype.h>
#include "thaBLAS.hpp"
#include "hip_helper.hpp"

// ----------------------------------------------------------------------------
// Transformer model
typedef struct {
  int dim; // transformer dimension
  int hidden_dim; // for ffn layers
  int n_layers; // number of layers
  int n_heads; // number of query heads
  int n_kv_heads; // number of key/value heads (can be < query heads because of multiquery)
  int vocab_size; // vocabulary size, usually 256 (byte-level)
  int seq_len; // max sequence length
} Config;

typedef struct {
  // token embedding table
  float* token_embedding_table;    // (vocab_size, dim)
  // weights for rmsnorms
  float* rms_att_weight; // (layer, dim) rmsnorm weights
  float* rms_ffn_weight; // (layer, dim)
  // weights for matmuls. note dim == n_heads * head_size
  float* wq; // (layer, dim, n_heads * head_size)
  float* wk; // (layer, dim, n_kv_heads * head_size)
  float* wv; // (layer, dim, n_kv_heads * head_size)
  float* wo; // (layer, n_heads * head_size, dim)
  // weights for ffn
  float* w1; // (layer, hidden_dim, dim)
  float* w2; // (layer, dim, hidden_dim)
  float* w3; // (layer, hidden_dim, dim)
  // final rmsnorm
  float* rms_final_weight; // (dim,)
  // (optional) classifier weights for the logits, on the last layer
  float* wcls;
} TransformerWeights;

typedef struct {
  // current wave of activations
  float *x; // activation at current time stamp (dim,)
  float *xb; // same, but inside a residual branch (dim,)
  float *xb2; // an additional buffer just for convenience (dim,)
  float *hb; // buffer for hidden dimension in the ffn (hidden_dim,)
  float *hb2; // buffer for hidden dimension in the ffn (hidden_dim,)
  float *q; // query (dim,)
  float *k; // key (dim,)
  float *v; // value (dim,)
  float *att; // buffer for scores/attention values (n_heads, seq_len)
  float *logits; // output logits
  // kv cache
  float* key_cache;   // (layer, seq_len, dim??) -> dim or kv_dim ??
  float* value_cache; // (layer, seq_len, dim??) -> dim or kv_dim ??
  float* key_matmul;
  float* value_matmul;
} RunState;

typedef struct {
  Config config; // the hyperparameters of the architecture (the blueprint)
  TransformerWeights weights; // the weights of the model
  RunState state; // buffers for the "wave" of activations in the forward pass
  // some more state needed to properly clean up the memory mapping (sigh)
  int fd; // file descriptor for memory mapping
  float* data; // memory mapped data pointer
  ssize_t file_size; // size of the checkpoint file in bytes
} Transformer;

void set_transformer();
void copy_transformer_to_device(thablasHandle_t handle, Transformer* t_h, Transformer* &t_d);
void copy_weight_to_device(Transformer* t_h, TransformerWeights* &w_d);
void alloc_state_to_device(Transformer* t_h, RunState* &s_d);
void alloc_state_to_device_batch(Transformer* t_h, RunState* &s_d_batch, int batch_size);
void copy_transformer_pipeline_to_device(thablasHandle_t handle, Transformer* t_h, Transformer* &t_d, int pipe_size, int pipe_id);
void copy_transformer_pipeline_to_device_batch(thablasHandle_t handle, Transformer* t_h, Transformer* &t_d, int pipe_size, int pipe_id, int batch_size);
// void copy_transformer_weight_pipeline_to_device_batch(thablasHandle_t handle, Transformer* t_h, TransformerWeights* &w_d, int pipe_size, int pipe_id, int batch_size);
// void alloc_run_state_to_device_batch(thablasHandle_t handle, Transformer* t_h, RunState* &s_d, int pipe_size, int pipe_id, int batch_size);
void free_transformer_device();


void scatter_transformer_weight_to_device(thablasHandle_t handle, Transformer* h_t, TransformerWeights* d_w, int d_id, int n_chunk_layers, int n_devices);
void alloc_run_state_to_device_batch(thablasHandle_t handle, Transformer* h_model, RunState* d_s, int n_chunk_layers, int batch_size);
