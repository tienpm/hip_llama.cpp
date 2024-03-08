#include <hip/hip_runtime.h>
#include <chrono>

// CHECK_HIP
#define CHECK_HIP(cmd) do {                                 \
  hipError_t error = cmd;                                   \
  if (error != hipSuccess) {                                \
    fprintf(stderr, "error: '%s'(%d) at %s:%d\n",           \
            hipGetErrorString(error), error,                \
            __FILE__, __LINE__);                             \
    exit(EXIT_FAILURE);                                     \
  }                                                         \
} while (0)

// template <unsigned int blockSize> 
__device__ void warpReduce6(volatile float *sdata, unsigned int tid) {
  int blockSize = blockDim.x;
  if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
  if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
  if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
  if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
  if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
  if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
}

// template <unsigned int blockSize>
__global__ void reduce6(float *g_idata, float *g_odata, unsigned int n) 
{
  const unsigned int blockSize = blockDim.x;
  extern __shared__ float sdata[];
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x*(blockSize*2) + tid;
  unsigned int gridSize = blockSize*2*gridDim.x;
  sdata[tid] = 0;
  while (i < n) { sdata[tid] += g_idata[i] + g_idata[i+blockSize]; i += gridSize; }
  __syncthreads();
  if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
  if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
  if (blockSize >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }
  if (tid < 32) warpReduce6(sdata, tid);
  if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

// modify  to caculate sum of squares

__device__ void warpReduce_v3(volatile float *sdata, unsigned int tid) {
  int blockSize = blockDim.x;
  // printf("blockSize: %d\n", blockSize);
  if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
  if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
  if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
  if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
  if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
  if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
}

__global__ void thaDNN_s_rmsnorm_kernel_v3(float* o, float* x, float* weight, int size)
{
  int n = size;
  float *g_idata = x;
  float *g_odata = o;
  const unsigned int blockSize = blockDim.x;
  extern __shared__ float sdata[];
  
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x*(blockSize*2) + tid;
  unsigned int gridSize = blockSize*2*gridDim.x;
  sdata[tid] = 0;
  while (i < n) { 
    sdata[tid] = sdata[tid] + g_idata[i]*g_idata[i]  + g_idata[i+blockSize]*g_idata[i+blockSize]; 
    i += gridSize; }

  __syncthreads();
  if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
  if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
  if (blockSize >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }
  if (tid < 32) warpReduce_v3(sdata, tid);
  if (tid == 0) g_odata[blockIdx.x] = sdata[0];
  
  __syncthreads();
  float ss = o[0];
  ss /= size;
  ss += 1e-5f;
  ss = 1.0f / sqrtf(ss);

  for(int i = blockIdx.x * blockDim.x + threadIdx.x; 
      i < size; 
      i += blockDim.x * gridDim.x) {
    o[i] = weight[i] * (ss * x[i]);
  }

}


// biến warpSize là biến built-in của HIP, mặc định là 64
__inline__ __device__
int warpReduceSum(int val) {
  // printf("warpSize: %d\n", warpSize);
  for (int offset = warpSize/2; offset > 0; offset /= 2) 
    val += __shfl_down(val, offset);
  return val;
}


__inline__ __device__
int blockReduceSum(int val) {

  static __shared__ int shared[64]; // Shared mem for 64 partial sums
  int lane = threadIdx.x % warpSize;
  int wid = threadIdx.x / warpSize;

  val = warpReduceSum(val);     // Each warp performs partial reduction

  if (lane==0) shared[wid]=val; // Write reduced value to shared memory

  __syncthreads();              // Wait for all partial reductions

  //read from shared memory only if that warp existed
  val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;

  if (wid==0) val = warpReduceSum(val); //Final reduce within first warp

  return val;
}


__global__ void deviceReduceBlockSum(int *in, int* out, int N) {
  int sum = int(0);
  for(int i = blockIdx.x * blockDim.x + threadIdx.x; 
      i < N; 
      i += blockDim.x * gridDim.x) {
    sum += in[i];
  }
  sum = blockReduceSum(sum);
  if (threadIdx.x == 0)
    atomicAdd(out, sum);
}

void deviceReduce1(int *in, int* out, int N) {
  int threads = 512;
  int blocks = min((N + threads - 1) / threads, 64*64);

  deviceReduceBlockSum<<<blocks, threads>>>(in, out, N);
  // deviceReduceSum<<<1, 1024>>>(out, out, blocks);
}

__global__ void deviceReduceWarpAtomicKernel(int *in, int* out, int N) {
  int sum = int(0);
  for(int i = blockIdx.x * blockDim.x + threadIdx.x; 
      i < N; 
      i += blockDim.x * gridDim.x) {
    sum += in[i];
  }
  sum = warpReduceSum(sum);
  if ((threadIdx.x & (warpSize - 1)) == 0)
    atomicAdd(out, sum);
}


void deviceReduce2(int *in, int* out, int N) {
  int threads = 512;
  int blocks = min((N + threads - 1) / threads, 64*64);

  deviceReduceWarpAtomicKernel<<<blocks, threads>>>(in, out, N);
  // deviceReduceSum<<<1, 1024>>>(out, out, blocks);
}


// modify deviceReduceBlockAtomicKernel to caculate sum of squares
__global__ void thaDNN_s_rmsnorm_kernel_v2_ok(float* o, float* x, float* weight, int size)
{
    int sum_squares = float(0);
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; 
        i < size; 
        i += blockDim.x * gridDim.x) {
        sum_squares += x[i] * x[i];
    }

    sum_squares = warpReduceSum(sum_squares);
    if ((threadIdx.x & (warpSize - 1)) == 0)
        atomicAdd(o, sum_squares);
    // syncthreads();
    __syncthreads();

    float ss = o[0];
    ss /= size;
    ss += 1e-5f;
    ss = 1.0f / sqrtf(ss);

    for(int i = blockIdx.x * blockDim.x + threadIdx.x; 
        i < size; 
        i += blockDim.x * gridDim.x) {
        o[i] = weight[i] * (ss * x[i]);
    }
}

// void test_rmsnorm_v2(float* o, float* x, float* weight, int size)
// {
//     dim3 blockDim(512);
//     dim3 gridDim(1);
//     thaDNN_s_rmsnorm_kernel_v2<<<gridDim, blockDim>>>(o, x, weight, size);
// }

// void test_rmsnorm_v3(float* o, float* x, float* weight, int size)
// {
//     dim3 blockDim(512);
//     dim3 gridDim(1);
//     thaDNN_s_rmsnorm_kernel_v3<<<gridDim, blockDim, 512*sizeof(float)>>>(o, x, weight, size);
// }


void rmsnorm(float* o, float* x, float* weight, int size) {
  // calculate sum of squares
  float ss = 0.0f;
  for (int j = 0; j < size; j++) {
    ss += x[j] * x[j];
  }
  ss /= size;
  ss += 1e-5f;
  ss = 1.0f / sqrtf(ss);
  // normalize and scale
  for (int j = 0; j < size; j++) {
    o[j] = weight[j] * (ss * x[j]);
  }
}


int main(){
  float *in, *out, *weight;
  float *out_cpu;
  float *in_h, *out_h, *weight_h;
  int LEN_ARRAY = 1024*16;
  in_h = (float *)malloc(LEN_ARRAY * sizeof(float));
  for (int i = 0; i < LEN_ARRAY; i++){
    in_h[i] = 0.1;
  }
  out_h = (float *)malloc(LEN_ARRAY * sizeof(float));

  weight_h = (float *)malloc(LEN_ARRAY * sizeof(float));
  for (int i = 0; i < LEN_ARRAY; i++){
    weight_h[i] = 2.5;
  }
    // check cpu
  out_cpu = (float *)malloc(LEN_ARRAY * sizeof(float));
  rmsnorm(out_cpu, in_h, weight_h, LEN_ARRAY);

  CHECK_HIP(hipMalloc((void**)&in, LEN_ARRAY*sizeof(float)));
  CHECK_HIP(hipMalloc((void**)&out, LEN_ARRAY*sizeof(float)));
  CHECK_HIP(hipMalloc((void**)&weight, LEN_ARRAY*sizeof(float)));
  // allocate memory on device for out elements

  CHECK_HIP(hipMemcpy(in, in_h, LEN_ARRAY*sizeof(float), hipMemcpyHostToDevice));
  CHECK_HIP(hipMemcpy(out, out_h, LEN_ARRAY*sizeof(float), hipMemcpyHostToDevice));
  CHECK_HIP(hipMemcpy(weight, weight_h, LEN_ARRAY*sizeof(float), hipMemcpyHostToDevice));
  
  // apply reduce6
  dim3 blockDim(512);
  dim3 gridDim(1);

  thaDNN_s_rmsnorm_kernel_v2_ok<<<gridDim, blockDim>>>(out, in, weight, LEN_ARRAY);
  // thaDNN_s_rmsnorm_kernel_v3<<<gridDim, blockDim, 512*sizeof(float)>>>(out, in, weight, LEN_ARRAY);
  // reduce6<<<gridDim, blockDim, 512*sizeof(float)>>>(in, out, LEN_ARRAY);
  //scpy result from device to host
  CHECK_HIP(hipMemcpy(out_h, out, LEN_ARRAY*sizeof(float), hipMemcpyDeviceToHost));
  CHECK_HIP(hipDeviceSynchronize());


  
  // compare result
  
  int cnt = 0 , thr = 10;
  float eps = 1e-5;
  for (int i = 0; i < LEN_ARRAY; ++i) {
    float o_gpu = out_h[i];
    float o_ans = out_cpu[i];
    if (fabsf(o_gpu - o_ans) > eps &&
        (o_ans == 0 || fabsf((o_gpu - o_ans) / o_ans) > eps)) {
      ++cnt;
      if (cnt <= thr)
        printf("O[%d] : correct_value = %f, your_value = %f\n", i, o_ans, o_gpu);
      if (cnt == thr + 1)
        printf("Too many error, only first %d values are printed.\n", thr);
    }
  }

  if (cnt == 0) {
    printf("Validation: VALID\n"); fflush(stdout);
  } else {
    printf("Validation: INVALID\n"); fflush(stdout);
  }

  return 0;
}
