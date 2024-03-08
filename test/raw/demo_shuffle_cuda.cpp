#include <hip/hip_runtime.h>


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

  static __shared__ int shared[32]; // Shared mem for 32 partial sums
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

__global__ void deviceReduceKernel(int *in, int* out, int N) {
  int sum = 0;
  //reduce multiple elements per thread
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
       i < N; 
       i += blockDim.x * gridDim.x) {
    sum += in[i];
  }
  sum = blockReduceSum(sum);
  if (threadIdx.x==0)
    out[blockIdx.x]=sum;
}

void deviceReduce(int *in, int* out, int N) {
  int threads = 512;
  int blocks = min((N + threads - 1) / threads, 1024);

  deviceReduceKernel<<<blocks, threads>>>(in, out, N);
  deviceReduceKernel<<<1, 1024>>>(out, out, blocks);
}

int main(){
  int *in, *out;
  int *in_h, *out_h;
  int LEN_ARRAY = 1024*4;
  in_h = (int *)malloc(LEN_ARRAY * sizeof(int));
  for (int i = 0; i < LEN_ARRAY; i++){
    in_h[i] = 1;
  }
  out_h = (int *)malloc(LEN_ARRAY * sizeof(int));
  for (int i = 0; i < LEN_ARRAY; i++){
    out_h[i] = 0;
  }

  CHECK_HIP(hipMalloc((void**)&in, LEN_ARRAY*sizeof(int)));
  CHECK_HIP(hipMalloc((void**)&out, LEN_ARRAY*sizeof(int)));
  // allocate memory on device for out elements

  CHECK_HIP(hipMemcpy(in, in_h, LEN_ARRAY*sizeof(int), hipMemcpyHostToDevice));
  CHECK_HIP(hipMemcpy(out, out_h, LEN_ARRAY*sizeof(int), hipMemcpyHostToDevice));

  deviceReduce(in, out, LEN_ARRAY);

  //prinft 5 fist value of out
  for (int i = 0; i < 18; i++) {
    printf("%d\n", out[i]);
  }
  return 0;
}
