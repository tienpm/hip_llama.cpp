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




template <unsigned int blockSize>
__device__ void warpReduce(volatile int *sdata, unsigned int tid) {
  if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
  if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
  if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
  if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
  if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
  if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
}

template <unsigned int blockSize>
__global__ void reduce6(int *g_idata, int *g_odata, unsigned int n) {
  extern __shared__ int sdata[];
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x*(blockSize*2) + tid;
  unsigned int gridSize = blockSize*2*gridDim.x;
  sdata[tid] = 0;
  while (i < n) { sdata[tid] += g_idata[i] + g_idata[i+blockSize]; i += gridSize; }
  __syncthreads();
  if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
  if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
  if (blockSize >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }
  if (tid < 32) warpReduce<blockSize>(sdata, tid);
  if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}


int main(){
  int *in, *out;
  int *in_h, *out_h;
  int LEN_ARRAY = 1024*512;
  in_h = (int *)malloc(LEN_ARRAY * sizeof(int));
  for (int i = 0; i < LEN_ARRAY; i++){
    in_h[i] = 1;
  }
  out_h = (int *)malloc(LEN_ARRAY * sizeof(int));
  // for (int i = 0; i < LEN_ARRAY; i++){
  //   out_h[i] = 0;
  // }

  CHECK_HIP(hipMalloc((void**)&in, LEN_ARRAY*sizeof(int)));
  CHECK_HIP(hipMalloc((void**)&out, LEN_ARRAY*sizeof(int)));
  // allocate memory on device for out elements

  CHECK_HIP(hipMemcpy(in, in_h, LEN_ARRAY*sizeof(int), hipMemcpyHostToDevice));
  CHECK_HIP(hipMemcpy(out, out_h, LEN_ARRAY*sizeof(int), hipMemcpyHostToDevice));

  deviceReduce2(in, out, LEN_ARRAY);




  std::chrono::time_point<std::chrono::system_clock> start, end;
  std::chrono::duration<double> elapsed_seconds;

  // start = std::chrono::system_clock::now();
  // int blockSize = 512;
  // int numBlocks = (LEN_ARRAY + blockSize - 1) / blockSize;
  // reduce6<512><<<numBlocks, blockSize, blockSize * sizeof(int)>>>(in, out, LEN_ARRAY);
  // end = std::chrono::system_clock::now();
  // elapsed_seconds = end - start;
  // printf("6 elapsed time: %f\n", elapsed_seconds.count());
  
  // start = std::chrono::system_clock::now();
  // deviceReduce1(in, out, LEN_ARRAY);
  // end = std::chrono::system_clock::now();
  // elapsed_seconds = end - start;
  // printf("1 elapsed time: %f\n", elapsed_seconds.count());


  

  start = std::chrono::system_clock::now();
  deviceReduce2(in, out, LEN_ARRAY);
  end = std::chrono::system_clock::now();
  elapsed_seconds = end - start;
  printf("2 elapsed time: %f\n", elapsed_seconds.count());


  
  //prinft 33 first value of out
  for (int i = 0; i < 3; i++) {
    printf("%d\n", out[i]);
  }
  return 0;
}
