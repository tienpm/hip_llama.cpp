#!/bin/bash

# hipcc -Wall -O1 -std=c++17 --offload-arch=gfx908 -lstdc++ -lm -fopenmp  test_mm.cpp -o test_mm_run
# srun -p EM  --gres=gpu:1 ./test_mm_run


hipcc -Wall -O1 -std=c++17 --offload-arch=gfx908  -lstdc++ -lm -fopenmp  test_tensor.cpp  ../src/thaBLAS.cpp ../src/thaDNN.cpp ../src/utils.cpp ../src/seq.cpp -I../include -I/opt/rocm/include/hipblas -o test_tensor_run
srun -p EM  --gres=gpu:1 ./test_tensor_run

# hipcc -Wall -O1 -std=c++17 --offload-arch=gfx908 -lstdc++ -lm -fopenmp thaBLAS.test.cpp ../src/thaBLAS.cpp                  ../src/utils.cpp ../src/seq.cpp -I../include -I/opt/rocm/include/hipblas -o test_thaBLAS_run
# hipcc -Wall -O1 -std=c++17 --offload-arch=gfx908 -lstdc++ -lm -fopenmp thaDNN.test.cpp ../src/thaBLAS.cpp ../src/thaDNN.cpp ../src/utils.cpp ../src/seq.cpp -I../include -I/opt/rocm/include/hipblas -o test_thaDNN_run

# srun -p EM  --gres=gpu:1 ./test_thaBLAS_run
# srun -p EM  --gres=gpu:1 ./test_thaDNN_run
# srun -p EM  --gres=gpu:4 ./test_thaBLAS_run
# srun -p EM  --gres=gpu:4 ./test_thaDNN_run
