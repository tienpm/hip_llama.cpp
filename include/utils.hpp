#include "hip_helper.hpp"

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fstream>
#include <iostream>

#include <hip/hip_runtime.h>

void alloc_mat(float **m, int R, int C) {
  // *m = (float *)aligned_alloc(32, sizeof(float) * R * C);
  CHECK_HIP(hipHostMalloc(m, R * C * sizeof(float)));
  if (*m == NULL) {
    printf("Failed to allocate memory for matrix.\n");
    exit(0);
  }
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

