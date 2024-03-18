#pragma once

#include "thaDNN.hpp"
#include "thaBLAS.hpp"
#include "hip_helper.hpp"

#include <hip/hip_runtime.h>

void initMemoryManager(MemoryManager *mm, int page_size, int row_size_in_bytes);

void initRunStatePaged(RunStatePaged* sp, int page_size, int row_size_in_bytes);

long long requestNewPage(MemoryManager *mm, int gid, float* &new_page_ptr);

void extendPage(RunStatePaged* sp, MemoryManager* mm, int gid, int pos);

float* getPage(RunStatePaged* sp, int pos);

void freeRunStatePaged(RunStatePaged* sp, MemoryManager* mm, int gid, int pos);
