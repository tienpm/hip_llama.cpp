#include <hip/hip_runtime.h>
#include <iostream>

#include "BatchManager.hpp"
#include "models.hpp"
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
  // K_data + V_data = 2 * kv_dim
  CHECK_HIP(hipMalloc(&block_addr, 2 * this->nrow_blocks * this->kv_dim * sizeof(float))); 
  this->logicId_physicAddr_mapper[new_block_id] = KVBlock{block_addr, 0};
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

KVBlock BatchManager::get_block_by_id(thablasHandle_t* handle, size_t block_id) {
  return this->logicId_physicAddr_mapper[block_id];
}

vector<KVBlock> BatchManager::convert_block_ids_to_block_addrs(thablasHandle_t* handle, vector<size_t> block_ids) {
  vector<KVBlock> block_addrs;
  for(size_t block_id: block_ids) {
    block_addrs.push_back(this->get_block_by_id(handle, block_id));
  }

}

void BatchManager::push_kv_data_to_kv_cache(thablasHandle_t* handle, vector<size_t> kv_cache_block_ids, float* data) {
  size_t last_block_id = kv_cache_block_ids.back();
  KVBlock last_block = this->get_block_by_id(handle, last_block_id);

  if (last_block.occupied >= this->nrow_blocks) {
    last_block_id = this->alloc_new_block(handle);
    kv_cache_block_ids.push_back(last_block_id);
    last_block = this->get_block_by_id(handle, last_block_id);
  }

  float* pos_to_push = last_block.addr + last_block.occupied * this->kv_dim;
  // K_data + V_data = 2 * kv_dim
  CHECK_HIP(hipMemcpyAsync(pos_to_push, data, 2 * kv_dim * sizeof(float), hipMemcpyDeviceToDevice, handle->calc_stream));
  ++last_block.occupied;
}

void BatchManager::dealloc_gpu_block(int seq_id) {

}

BatchManager::~BatchManager() {

}
