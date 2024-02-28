#!/bin/bash

# gemm
# hipcc -Wall -O1 -std=c++17 --offload-arch=gfx908 -lstdc++ -lm -fopenmp thaBLAS.test.cpp ../src/thaBLAS.cpp -I../include -I/opt/rocm/include/hipblas -o test_run
# srun -p EM  --gres=gpu:4 ./test_run

# sum vector elements
hipcc -Wall -O1 -std=c++17 --offload-arch=gfx908 -lstdc++ -lm -fopenmp thaBLASSoftmax.test.cpp ../src/thaBLAS.cpp -I../include -I/opt/rocm/include/hipblas -o test_run
