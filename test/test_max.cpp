#include <stdio.h>

#include "hip/hip_runtime.h"

const float float_min = -3.402e+38;

// define CHECK_HIP macro with inline
#define CHECK_HIP(cmd)                                                         \
  do {                                                                         \
    hipError_t error = cmd;                                                    \
    if (error != hipSuccess) {                                                 \
      fprintf(stderr, "error: '%s'(%d) at %s:%d\n", hipGetErrorString(error),   \
              error, __FILE__, __LINE__);                                      \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

// __global__ void maxReduce(volatile float* d_data, int n)
// {
//     int ti = hipThreadIdx_x;

//     __shared__ volatile float max_value;

//     if (ti == 0) max_value = d_float_min;

//     for (int bi = 0; bi < n; bi += 32)
//     {
//         int i = bi + ti;
//         if (i >= n) break;

//         float v = d_data[i];
//         __syncthreads();

//         while (max_value < v)
//         {
//             max_value = v;
//         }

//         __syncthreads();
//     }

//     if (ti == 0) d_data[0] = max_value;
// }

#define WARP_SIZE 64
#define MAX_BLOCK_SIZE 1024

// __device__ float warp_reduce_sum(float val)

//     for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
//         val += __shfl_xor(val, offset);
//     return val;
// }

// __device__ float block_reduce_sum(float val)
// {
//     static __shared__ float shared[MAX_BLOCK_SIZE / WARP_SIZE];
//     int lane = threadIdx.x % WARP_SIZE;
//     int wid = threadIdx.x / WARP_SIZE;

//     val = warp_reduce_sum(val);

//     if (lane == 0)
//         shared[wid] = val;

//     __syncthreads();

//     val = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared[lane] : 0;

//     if (wid == 0)
//         val = warp_reduce_sum(val);

//     return val;
// }

// find max value (for numerical stability)
__device__ float warp_reduce_max(float val) {
  for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
    val = max(val, __shfl_xor(val, offset));
  return val;
}

__device__ float block_reduce_max(float val) {
  static __shared__ float shared[32];
  int lane = threadIdx.x % WARP_SIZE;
  int wid = threadIdx.x / WARP_SIZE;

  val = warp_reduce_max(val);

  if (lane == 0) shared[wid] = val;

  __syncthreads();

  val = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared[lane] : -3.402e+38;

  if (wid == 0) val = warp_reduce_max(val);

  return val;
}

__global__ void maxReduce(volatile float* data, int n, float *max_value_return) {
  int lane_x = threadIdx.x;
  __shared__ float max_value;

  float val = -3.402e+38;

  for (int i = lane_x; i < n; i += blockDim.x) {
    val = max(val, data[i]);
  }

  val = block_reduce_max(val);

  if (lane_x == 0) max_value = val;

  __syncthreads();

  if (blockIdx.x == 0 && threadIdx.x == 0) max_value_return[0] = max_value;
}

void testMax(int n, bool verbose) {
  float *h_data, *d_data;
  float cpu_max = float_min;
  float *max_value_return;
  float *h_max_value_return;

  // allocate memory on the cpu by hipHostMalloc
  CHECK_HIP(hipHostMalloc(&h_data, n * sizeof(float), hipHostMallocDefault));
  CHECK_HIP(hipHostMalloc(&h_max_value_return, sizeof(float), hipHostMallocDefault));

  // allocate memory on the gpu
  CHECK_HIP(hipMalloc(&d_data, n * sizeof(float)));
  CHECK_HIP(hipMalloc(&max_value_return, sizeof(float)));

  // initialize the input data on the cpu 
  for (int i = 0; i < n; i++) {
    // random flaot number between -100 to 100 
    h_data[i] = (float) (rand()  - 100);
  }

  printf("h_data[0] = %f\n", h_data[0]);
  printf("h_data[1] = %f\n", h_data[1]);
  printf("h_data[2] = %f\n", h_data[2]);
  // copy data to the gpu
  CHECK_HIP(hipMemcpy(d_data, h_data, n * sizeof(float), hipMemcpyHostToDevice));

  // do the max on the cpu
  for (int i = 0; i < n; i++) {
    
    if (h_data[i] > cpu_max) {
      cpu_max = h_data[i];
    }
    // printf("h_data[%d] = %f\n", i, h_data[i]);
  }

  // do the max on the gpu
  maxReduce<<<1, 256>>>(d_data, n, max_value_return);

  CHECK_HIP(hipMemcpy(h_max_value_return, max_value_return, sizeof(float), hipMemcpyDeviceToHost));

  // free memory on host and device
  // free(h_data);

  // did the gpu get the same answer as the cpu?
  if (verbose)
  {
    if (cpu_max != h_max_value_return[0])
        {printf("n =%6d cpu_max = %f, gpu_max = %f, FAIL  ", n, cpu_max, h_max_value_return[0]);
         }
    else
    {
    printf("n =%6d cpu_max = %f, gpu_max = %f, PASS  ", n, cpu_max, h_max_value_return[0]);
    }
  }

  // copy d_data to h_data
  CHECK_HIP(hipMemcpy(h_data, d_data, n * sizeof(float), hipMemcpyDeviceToHost));
  printf("h_data[0] = %f\n", h_data[0]);
  printf("h_data[1] = %f\n", h_data[1]);
  printf("h_data[2] = %f\n", h_data[2]);

}

/*
* softmax
*/

// void softmax(float* x, int size) {
//   // find max value (for numerical stability)
//   float max_val = x[0];
//   for (int i = 1; i < size; i++) {
//     if (x[i] > max_val) {
//       max_val = x[i];
//     }
//   }
//   // exp and sum
//   float sum = 0.0f;
//   for (int i = 0; i < size; i++) {
//     x[i] = expf(x[i] - max_val);
//     sum += x[i];
//   }
//   // normalize
//   for (int i = 0; i < size; i++) {
//     x[i] /= sum;
//   }
// }

// explosive x[i] - max_val and reduce the sum



__device__ float maxReduce_device(volatile float* data, int n) {
  int lane_x = threadIdx.x;
  __shared__ float max_value;

  float val = -3.402e+38;

  for (int i = lane_x; i < n; i += blockDim.x) {
    val = max(val, data[i]);
  }

  val = block_reduce_max(val);

  if (lane_x == 0) max_value = val;

  __syncthreads();

  // if (blockIdx.x == 0 && threadIdx.x == 0) max_value_return[0] = max_value;
  return max_value;
}

// biến warpSize là biến built-in của HIP, mặc định là 64

__device__ float warp_reduce_sum(float val)
{
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) 
        val += __shfl_xor(val, offset);
    return val;
}

__device__ float block_reduce_sum(float val) 
{
    static __shared__ float shared[MAX_BLOCK_SIZE / WARP_SIZE]; 
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;

    val = warp_reduce_sum(val); 

    if (lane == 0)
        shared[wid] = val; 

    __syncthreads(); 

    val = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared[lane] : 0;

    if (wid == 0)
        val = warp_reduce_sum(val); 

    return val;
}

__global__ void thaDNN_s_softmax_kernel_v2(float* x, int size)
{
  /*
  * reduction to find max value 
  */
  float max_value_return = -3.402e+38;
  max_value_return = maxReduce_device(x, size);
  
  for (int i = threadIdx.x; i < size; i += blockDim.x) {
    x[i] = expf(x[i] - max_value_return);
  }

  float ss = 0.0f;
  // __shared__ float total_sum;
  for (int i = threadIdx.x; i < size; i += blockDim.x) {
    ss += x[i];
  }
  ss = block_reduce_sum(ss);

  for (int i = threadIdx.x; i < size; i += blockDim.x) {
    x[i] /= ss;
  }
}

void softmax(float* x, int size) {
  // find max value (for numerical stability)
  float max_val = x[0];
  for (int i = 1; i < size; i++) {
    if (x[i] > max_val) {
      max_val = x[i];
    }
  }
  // exp and sum
  float sum = 0.0f;
  for (int i = 0; i < size; i++) {
    x[i] = expf(x[i] - max_val);
    sum += x[i];
  }
  // normalize
  for (int i = 0; i < size; i++) {
    x[i] /= sum;
  }
}



void testSoftMax(int n, bool verbose) {
  float *h_data, *d_data;
  float cpu_max = float_min;
  float *max_value_return;
  float *h_max_value_return;

  // allocate memory on the cpu by hipHostMalloc
  CHECK_HIP(hipHostMalloc(&h_data, n * sizeof(float), hipHostMallocDefault));
  CHECK_HIP(hipHostMalloc(&h_max_value_return, sizeof(float), hipHostMallocDefault));

  // allocate memory on the gpu
  CHECK_HIP(hipMalloc(&d_data, n * sizeof(float)));
  CHECK_HIP(hipMalloc(&max_value_return, sizeof(float)));

  // initialize the input data on the cpu 
  for (int i = 0; i < n; i++) {
    // random flaot number between 0 to 1
    h_data[i] = (float) (rand() % 100) / 100;
  }

  // copy h_data to a new array h_data_check
  float *h_data_check = (float *)malloc(n * sizeof(float));
  for (int i = 0; i < n; i++) {
    h_data_check[i] = h_data[i];
  }

  // copy data to the gpu
  CHECK_HIP(hipMemcpy(d_data, h_data, n * sizeof(float), hipMemcpyHostToDevice));

  // do the max on the cpu
  for (int i = 0; i < n; i++) {
    
    if (h_data[i] > cpu_max) {
      cpu_max = h_data[i];
    }
    // printf("h_data[%d] = %f\n", i, h_data[i]);
  }

  // do the max on the gpu
  thaDNN_s_softmax_kernel_v2<<<1, 256>>>(d_data, n);

  CHECK_HIP(hipMemcpy(h_data, d_data, n * sizeof(float), hipMemcpyDeviceToHost));

  // check the result
  softmax(h_data_check, n);

  // did the gpu get the same answer as the cpu?
  if (verbose)
  {
    int count = 0;
    int max_count = 10;
    for (int i = 0; i < n; i++) {
      if (h_data[i] != h_data_check[i]) {
        count++;
        if (count < max_count) {
          printf("h_data[%d] = %f, h_data_check[%d] = %f\n", i, h_data[i], i, h_data_check[i]);
        }
      }
    }
  }
}



int main() {

  // testMax(1000000, false);
  testSoftMax(1000, true);

  return 0;
}
