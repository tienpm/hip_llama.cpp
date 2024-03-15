#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>

#include <fstream>
#include <iostream>

#include <hip/hip_runtime.h>
#include "hip_helper.hpp"
#include "thaBLAS.hpp"
#include "models.hpp"
#include "utils.hpp"

void alloc_mat(float **m, int R, int C) {
  // *m = (float *)aligned_alloc(32, sizeof(float) * R * C);
  // ---!!! CAUTION !!!--- Remember to replace hipFree with free
  CHECK_HIP(hipHostMalloc(m, R * C * sizeof(float)));
  if (*m == NULL) {
    printf("Failed to allocate memory for matrix.\n");
    exit(0);
  }
}

void util_free(void *m)
{
  // free(m);
  CHECK_HIP(hipFree(m));
}

void alloc_vec(float **m, int N) {
  alloc_mat(m, N, 1);
}

void rand_mat(float *m, int R, int C) {
  for (int i = 0; i < R; i++) {
    for (int j = 0; j < C; j++) {
      m[i * C + j] = (float)rand() / (float)RAND_MAX - 0.5;
    }
  }
}

void rand_vec(float *m, int N) {
  rand_mat(m, N, 1);
}

void zero_mat(float *m, int R, int C) { memset(m, 0, sizeof(float) * R * C); }

void zero_vec(float *m, int N) { zero_mat(m, N, 1); }

bool compareFiles(const std::string& filePath1, const std::string& filePath2) {
    std::ifstream file1(filePath1, std::ifstream::binary | std::ifstream::ate);
    std::ifstream file2(filePath2, std::ifstream::binary | std::ifstream::ate);

    if (!file1.is_open() || !file2.is_open()) {
        std::cerr << "Error opening one of the files for comparison." << std::endl;
        return false;
    }

    // Check file sizes first
    if (file1.tellg() != file2.tellg()) {
        std::cerr << "Files have different sizes." << std::endl;
        return false;
    }

    // Reset file read positions
    file1.seekg(0, std::ifstream::beg);
    file2.seekg(0, std::ifstream::beg);

    char buffer1, buffer2;
    while (file1.get(buffer1) && file2.get(buffer2)) {
        if (buffer1 != buffer2) {
            return false; 
        }
    }

    return true; 
}

void malloc_run_state(RunState* s, Config* p) {
  // we calloc instead of malloc to keep valgrind happy
  int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
  s->x = (float*)calloc(p->dim, sizeof(float));
  s->xb = (float*)calloc(p->dim, sizeof(float));
  s->xb2 = (float*)calloc(p->dim, sizeof(float));
  s->hb = (float*)calloc(p->hidden_dim, sizeof(float));
  s->hb2 = (float*)calloc(p->hidden_dim, sizeof(float));
  s->q = (float*)calloc(p->dim, sizeof(float));
  s->key_cache = (float*)calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
  s->value_cache = (float*)calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
  s->att = (float*)calloc(p->n_heads * p->seq_len, sizeof(float));
  s->logits = (float*)calloc(p->vocab_size, sizeof(float));
  // ensure all mallocs went fine
  if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->q
      || !s->key_cache || !s->value_cache || !s->att || !s->logits) {
    fprintf(stderr, "malloc failed!\n");
    exit(EXIT_FAILURE);
  }
}

void print_transformer(Transformer* t) {
  printf("---------Model Information----------\n");
  printf("dim: %d\n", t->config.dim);
  printf("hidden_dim: %d\n", t->config.hidden_dim);
  printf("n_layers: %d\n", t->config.n_layers);
  printf("n_heads: %d\n", t->config.n_heads);
  printf("n_kv_heads: %d\n", t->config.n_kv_heads);
  printf("vocab_size: %d\n", t->config.vocab_size);
  printf("seq_len: %d\n", t->config.seq_len);
  printf("weights_size: %lu MB\n", (t->file_size - sizeof(Config)) / (1024L*1024L));
  printf("------------------------------------\n");
}

void memory_map_weights(TransformerWeights *w, Config* p, float* ptr, int shared_weights) {
  int head_size = p->dim / p->n_heads;
  // make sure the multiplications below are done in 64bit to fit the parameter counts of 13B+ models
  unsigned long long n_layers = p->n_layers;
  w->token_embedding_table = ptr;
  ptr += p->vocab_size * p->dim;
  w->rms_att_weight = ptr;
  ptr += n_layers * p->dim;
  w->wq = ptr;
  ptr += n_layers * p->dim * (p->n_heads * head_size);
  w->wk = ptr;
  ptr += n_layers * p->dim * (p->n_kv_heads * head_size);
  w->wv = ptr;
  ptr += n_layers * p->dim * (p->n_kv_heads * head_size);
  w->wo = ptr;
  ptr += n_layers * (p->n_heads * head_size) * p->dim;
  w->rms_ffn_weight = ptr;
  ptr += n_layers * p->dim;
  w->w1 = ptr;
  ptr += n_layers * p->dim * p->hidden_dim;
  w->w2 = ptr;
  ptr += n_layers * p->hidden_dim * p->dim;
  w->w3 = ptr;
  ptr += n_layers * p->dim * p->hidden_dim;
  w->rms_final_weight = ptr;
  ptr += p->dim;
  ptr += p->seq_len * head_size / 2; // skip what used to be freq_cis_real (for RoPE)
  ptr += p->seq_len * head_size / 2; // skip what used to be freq_cis_imag (for RoPE)
  w->wcls = shared_weights ? w->token_embedding_table : ptr;
}

void read_checkpoint(char* checkpoint, Config* config, TransformerWeights* weights,
    int* fd, float** data, ssize_t* file_size) {
  FILE *file = fopen(checkpoint, "rb");
  if (!file) { fprintf(stderr, "Couldn't open file %s\n", checkpoint); exit(EXIT_FAILURE); }
  // read in the config header
  if (fread(config, sizeof(Config), 1, file) != 1) { exit(EXIT_FAILURE); }
  // negative vocab size is hacky way of signaling unshared weights. bit yikes.
  int shared_weights = config->vocab_size > 0 ? 1 : 0;
  config->vocab_size = abs(config->vocab_size);
  // figure out the file size
  fseek(file, 0, SEEK_END); // move file pointer to end of file
  *file_size = ftell(file); // get the file size, in bytes
  fclose(file);
  // memory map the Transformer weights into the data pointer
  *fd = open(checkpoint, O_RDONLY); // open in read only mode
  if (*fd == -1) { fprintf(stderr, "open failed!\n"); exit(EXIT_FAILURE); }
  *data = (float *)mmap(NULL, *file_size, PROT_READ, MAP_PRIVATE, *fd, 0);
  if (*data == MAP_FAILED) { fprintf(stderr, "mmap failed!\n"); exit(EXIT_FAILURE); }
  float* weights_ptr = *data + sizeof(Config)/sizeof(float);
  memory_map_weights(weights, config, weights_ptr, shared_weights);
}

void build_transformer(Transformer *t, char* checkpoint_path) {
  // read in the Config and the Weights from the checkpoint
  read_checkpoint(checkpoint_path, &t->config, &t->weights, &t->fd, &t->data, &t->file_size);
  // allocate the RunState buffers
  malloc_run_state(&t->state, &t->config);
  print_transformer(t);
}

void free_run_state(RunState* s) {
  free(s->x);
  free(s->xb);
  free(s->xb2);
  free(s->hb);
  free(s->hb2);
  free(s->q);
  free(s->att);
  free(s->logits);
  free(s->key_cache);
  free(s->value_cache);
}

void free_transformer(Transformer* t) {
  // close the memory mapping
  if (t->data != MAP_FAILED) { munmap(t->data, t->file_size); }
  if (t->fd != -1) { close(t->fd); }
  // free the RunState buffers
  free_run_state(&t->state);
}

/*
 *      UTILS ON GPU
 * */

// copy transformer checkpoint data from host to device
// all scalar values and pointer values are still stored on host
// only arrays are stored on device
void copy_transformer_to_device(thablasHandle_t handle, Transformer* t_h, Transformer* &t_d)
{
  CHECK_HIP(hipSetDevice(handle.current_gpu_id));

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
  int n_layers = layer;
  int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
  int hidden_dim = p->hidden_dim;
  int n_heads = p->n_heads;
  int seq_len = p->seq_len;

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

void alloc_state_to_device_batch(Transformer* t_h, RunState* &s_d_batch, int n_batches)
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

  CHECK_HIP(hipMalloc(&s_d_batch->x, dim * sizeof(float) * n_batches));
  CHECK_HIP(hipMalloc(&s_d_batch->xb, dim * sizeof(float) * n_batches));
  CHECK_HIP(hipMalloc(&s_d_batch->xb2, dim * sizeof(float) * n_batches));
  CHECK_HIP(hipMalloc(&s_d_batch->hb, hidden_dim * sizeof(float) * n_batches));
  CHECK_HIP(hipMalloc(&s_d_batch->hb2, hidden_dim * sizeof(float) * n_batches));
  CHECK_HIP(hipMalloc(&s_d_batch->q, dim * sizeof(float) * n_batches));
  CHECK_HIP(hipMalloc(&s_d_batch->att, n_heads * seq_len * sizeof(float) * n_batches));
  CHECK_HIP(hipMalloc(&s_d_batch->logits, vocab_size * sizeof(float) * n_batches));
  CHECK_HIP(hipMalloc(&s_d_batch->key_cache, n_layers * seq_len * kv_dim * sizeof(float) * n_batches));
  CHECK_HIP(hipMalloc(&s_d_batch->value_cache, n_layers * seq_len * kv_dim * sizeof(float) * n_batches));
}

void copy_transformer_pipeline_to_device(thablasHandle_t handle, Transformer* t_h, Transformer* &t_d, int pipe_size, int pipe_id)
{
  CHECK_HIP(hipSetDevice(handle.current_gpu_id));
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

void free_device_run_state() {

}

void free_device_transformer() {

}
