#!/bin/bash


# hipcc -Wall -O1 -std=c++17 --offload-arch=gfx908  -L/opt/rocm/include/rocblas/ -lrocblas -fopenmp -mavx2 test_roc.cpp -o test_roc && srun -p EM  --gres=gpu:1 ./test_roc

hipcc -Wall -O1 -std=c++17 --offload-arch=gfx908  -L/opt/rocm/include/rocblas/ -lrocblas -fopenmp -mavx2 check.cpp -o check && srun -p EM  --gres=gpu:1 ./check


# hipcc -Wall -O1 -std=c++17 --offload-arch=gfx908  -L/opt/rocm/include/rocblas/ -lrocblas -fopenmp -mavx2 check_gemm.cpp -o check_gemm
# srun -p EM  --gres=gpu:1 ./check_gemm


# hipcc -Wall -O1 -std=c++17 --offload-arch=gfx908 -lstdc++ -lm -fopenmp thaBLAS.test.cpp ../src/thaBLAS.cpp                                    ../src/utils.cpp ../src/seq.cpp -I../include -I/opt/rocm/include/hipblas -o test_thaBLAS_run
# hipcc -Wall -O1 -std=c++17 --offload-arch=gfx908 -lstdc++ -lm -fopenmp thaDNN.test.cpp ../src/thaBLAS.cpp ../src/thaDNN.cpp ../src/thaDNN_batch.cpp ../src/utils.cpp ../src/seq.cpp -I../include -I/opt/rocm/include/hipblas -o test_thaDNN_run

# srun -p EM  --gres=gpu:1 ./test_thaBLAS_run
# srun -p EM  --gres=gpu:1 ./test_thaDNN_run
# srun -p EM  --gres=gpu:4 ./test_thaBLAS_run
# srun -p EM  --gres=gpu:4 ./test_thaDNN_run
