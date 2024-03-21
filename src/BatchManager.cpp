#include <hip/hip_runtime.h>
#include <iostream>

#include "BatchManager.hpp"
#include "thaBLAS.hpp"


BatchManager::BatchManager(int batch_size, int seq_len, int n_layers, int kv_dim) {
  this->batch_size = batch_size;
  this->seq_len = seq_len;
  this->n_layers = n_layers;
  this->kv_dim = kv_dim;
}

size_t BatchManager::alloc_new_block(thablasHandle_t* handle) {
  int new_block_id = ++this->cur_physical_blocks;
  CHECK_HIP(hipSetDevice(handle->current_gpu_id));

  float* block_addr;
  CHECK_HIP(hipMalloc(&block_addr, this->nrow_blocks * this->kv_dim * sizeof(float)));
  this->logicId_physicAddr_mapper[new_block_id] = make_pair(block_addr, 0);
  return new_block_id;
}

void BatchManager::set_gpu_physical_blocks(int d_id) {
  
}

void BatchManager::set_gpu_block() {

}

size_t BatchManager::get_gpu_memory(int d_id) {
  CHECK_HIP(hipSetDevice(d_id));       
  size_t freeMem, totalMem;
  CHECK_HIP(hipMemGetInfo(&freeMem, &totalMem));

  return freeMem;
}

pair<float*, int> BatchManager::get_physicAddr_from_id(thablasHandle_t* handle, size_t block_id) {
  return this->logicId_physicAddr_mapper[block_id];
}

vector<pair<float*, int>> BatchManager::convert_block_ids_to_block_addrs(thablasHandle_t* handle, vector<size_t> block_ids) {
  vector<pair<float*, int>> block_addrs;
  for(size_t block_id: block_ids) {
    block_addrs.push_back(this->get_physicAddr_from_id(handle, block_id));
  }

}

void BatchManager::push_kv_data_to_kv_cache(thablasHandle_t* handle, vector<size_t> kv_cache_block_ids, float* data) {
  size_t last_block_id = kv_cache_block_ids.back();
  pair<float*, int> last_block = this->get_physicAddr_from_id(handle, last_block_id);

  if (last_block.second >= this->nrow_blocks)

  if (last_block.second < this->nrow_blocks)
  {
    float* pos_to_push = last_block.first + last_block.second * this->kv_dim;
    ++last_block.second;

    CHECK_HIP(hipMemcpyAsync(pos_to_push, data, kv_dim * sizeof(float), hipMemcpyDeviceToDevice, handle->calc_stream));
  }


}

void BatchManager::dealloc_gpu_block(int seq_id) {

}

BatchManager::~BatchManager() {

}
