#!/bin/bash

hipcc -Wall -O3 -std=c++17 --offload-arch=gfx908 -lstdc++ -lm -fopenmp test_thaBLAS_s_matmul.cpp ../src/thaBLAS.cpp ../src/utils.cpp -I../include -I/opt/rocm/include/hipblas -o run_matmul_test
