#pragma once
#include <vector>
#include <map>

#include "hip_helper.hpp"
#include "thaBLAS.hpp"

using namespace std;

class BatchManager {
public:
  size_t num_physical_blocks;
  size_t cur_physical_blocks;
  int block_size;
  int seq_len;
  int batch_size;
  int n_layers;
  int kv_dim;
  int nrow_blocks;

  vector<pair<int, int>> logical_KV_table;
  vector<int> n_block_table_filled;
  vector<vector<bool>> mask_physical_KV_block;
  float* gpu_physical_KV_blocks;
  vector<float*> cpu_physical_KV_blocks;

  map<size_t, pair<float*, int> > logicId_physicAddr_mapper; 


  BatchManager(int batch_size, int seq_len, int n_layers, int kv_dim);

  void set_gpu_physical_blocks(int d_id);

  void set_gpu_block();

  size_t get_gpu_memory(int d_id);

  void get_KV_Block(int seq_id);

  size_t alloc_new_block(thablasHandle_t* handle);

  pair<float*, int> get_physicAddr_from_id(thablasHandle_t* handle, size_t seq_id);

  vector<pair<float*, int>> convert_block_ids_to_block_addrs(thablasHandle_t* handle, vector<size_t> block_ids);

  void push_kv_data_to_kv_cache(thablasHandle_t* handle, vector<size_t> kv_cache_block_ids, float* data);

  void dealloc_gpu_block(int seq_id);

  ~BatchManager();
};

