#pragma once
#include <vector>
#include <map>

#include "hip_helper.hpp"

using namespace std;

class BatchManager {
public:
  int num_physical_blocks;
  int cur_physical_blocks;
  int block_size;
  int seq_len;
  int batch_size;
  int n_layers;
  int nrow_blocks;
  int kv_dim;

  vector<pair<int, int>> logical_KV_table;
  vector<int> n_block_table_filled;
  vector<vector<bool>> mask_physical_KV_block;
  float* gpu_physical_KV_blocks;
  vector<float*> cpu_physical_KV_blocks;

  BatchManager(int batch_size, int seq_len, int n_layers, int kv_dim);

  void set_gpu_physical_blocks(int d_id);

  void set_gpu_block();

  size_t get_gpu_memory(int d_id);

  void get_KV_Block(int seq_id);

  void dealloc_gpu_block(int seq_id);
  
  ~BatchManager();
};

