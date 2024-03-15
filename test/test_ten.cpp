#include <hip/hip_runtime.h>
#define M 16
#define N 16
#define K 4

#define CHECK_HIP(x)                                \
  if (x != hipSuccess) {                            \
    printf("Error at %s:%d\n", __FILE__, __LINE__); \
    return;                                         \
  }

__global__ void segmm_16x16x4( float *A, float *B, float *D) {
  using float4 = __attribute__((__vector_size__(K * sizeof(float)))) float;

  float4 dmn = {0};

  int mk = threadIdx.y + K * threadIdx.x;
  int kn = threadIdx.x + N * threadIdx.y;

  float amk = A[mk];
  float bkn = B[kn];

  dmn = __builtin_amdgcn_mfma_f32_16x16x4f32(amk, bkn, dmn, 0, 0, 0);

  for (int i = 0; i < 4; i++) {
    const int idx = threadIdx.x + i * N + threadIdx.y * 4 * N;
    D[idx] = dmn[i];
  }
}

void matmul_cpu(float *A, float *B, float *C)
{
  for (int i = 0; i < M; ++i) {
    for (int k = 0; k < K; ++k) {
      for (int j = 0; j < N; ++j) {
        C[i * N + j] += A[i * K + k] * B[k * N + j];
      }
    }
  }
}


void test() {
  float *A, *B, *C, *C_ans;
  hipMalloc(&A, M * K * sizeof(float));
  hipMalloc(&B, K * N * sizeof(float));
  hipMalloc(&C, M * N * sizeof(float));
  hipMalloc(&C_ans, M * N * sizeof(float));

  float *A_h = (float *)malloc(M * K * sizeof(float));
  float *B_h = (float *)malloc(K * N * sizeof(float));
  float *C_h = (float *)malloc(M * N * sizeof(float));
  float *C_ans_h = (float *)malloc(M * N * sizeof(float));

  for (int i = 0; i < M * K; i++) A_h[i] = 0.1f;
  for (int i = 0; i < K * N; i++) B_h[i] = 0.2f;
  // for (int i = 0; i < M * N; i++) C_h[i] = 0.0f;

  hipMemcpy(A, A_h, M * K * sizeof(float), hipMemcpyHostToDevice);
  hipMemcpy(B, B_h, K * N * sizeof(float), hipMemcpyHostToDevice);
  // hipMemcpy(C, C_h, M * N * sizeof(float), hipMemcpyHostToDevice);

  matmul_cpu(A_h, B_h, C_ans_h);
  hipMemcpy(C_ans, C_ans_h, M * N * sizeof(float), hipMemcpyHostToDevice);
  hipDeviceSynchronize();


  dim3 grid(1, 1, 1);
  dim3 block(16, 4, 1);
  segmm_16x16x4<<<grid, block>>>(A, B, C);

  hipMemcpy(C_h, C, M * N * sizeof(float), hipMemcpyDeviceToHost);
  hipDeviceSynchronize();

  bool is_valid = true;
  int cnt = 0, thr = 100;
  float eps = 1e-4;
  for (int i = 0; i < M * N; i++) {
    float c = C_h[i];
    float c_ans = C_ans_h[i];
    if (fabsf(c - c_ans) > eps &&
        (c_ans == 0 || fabsf((c - c_ans) / c_ans) > eps)) {
      ++cnt;
      if (cnt <= thr)
        printf("C[%d] : correct_value = %f, your_value = %f\n", i, c, c_ans);
      if (cnt == thr + 1)
        printf("Too many error, only first %d values are printed.\n", thr);
      is_valid = false;
    }
  }

  if (is_valid) {
    printf("Validation: VALID\n");
  } else {
    printf("Validation: INVALID\n");
  }
}

int main() {
  test();
  return 0;
}
