#include "hip_helper.hpp"
#include "models.hpp"

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
