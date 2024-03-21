#include <hip/hip_runtime.h>
#include <iostream>

#include "BatchManager.hpp"


BatchManager::BatchManager(int batch_size, int seq_len, int n_layers, int kv_dim) {
  this->batch_size = batch_size;
  this->seq_len = seq_len;
  this->n_layers = n_layers;
  this->kv_dim = kv_dim;
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

void BatchManager::get_KV_Block(int seq_id) {

}

void BatchManager::dealloc_gpu_block(int seq_id) {

}

BatchManager::~BatchManager() {

}
