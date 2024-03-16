#include "hip_helper.hpp"
#include "thaBLAS.hpp"
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

void alloc_mat(float **m, int R, int C);

void util_free(void *m);

void alloc_vec(float **m, int N);

void rand_mat(float *m, int R, int C);

void rand_vec(float *m, int N);

void zero_mat(float *m, int R, int C);

void zero_vec(float *m, int N);

bool compareFiles(const std::string& filePath1, const std::string& filePath2);

/*
 * ===========================================================================
 *    Initial Transformer 
 * ===========================================================================
 */

void malloc_run_state(RunState* s, Config* p);

void print_transformer(Transformer* t);

void memory_map_weights(TransformerWeights *w, Config* p, float* ptr, int shared_weights);

void read_checkpoint(char* checkpoint, Config* config, TransformerWeights* weights, int* fd, float** data, ssize_t* file_size);

void build_transformer(Transformer *t, char* checkpoint_path);

void free_run_state(RunState* s);

void free_transformer(Transformer* t); 
// ON GPU functions

void copy_transformer_to_device(thablasHandle_t handle, Transformer* t_h, Transformer* &t_d);

void copy_weight_to_device(Transformer* t_h, TransformerWeights* &w_d);

void alloc_state_to_device(Transformer* t_h, RunState* &s_d);

void alloc_state_to_device_batch(Transformer* t_h, RunState* &s_d_batch, int n_batches);

void copy_transformer_pipeline_to_device(thablasHandle_t handle, Transformer* t_h, Transformer* &t_d, int pipe_size, int pipe_id);

void copy_transformer_pipeline_to_device_batch(thablasHandle_t handle, Transformer* t_h, Transformer* &t_d, int pipe_size, int pipe_id, int n_batches);

void copy_transformer_weight_pipeline_to_device_batch(thablasHandle_t handle, Transformer* t_h, TransformerWeights* &w_d, int pipe_size, int pipe_id, int batch_size);

void alloc_run_state_to_device_batch(thablasHandle_t handle, Transformer* t_h, RunState* &s_d, int pipe_size, int pipe_id, int batch_size);
