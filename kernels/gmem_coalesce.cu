#include "utils/common.cuh"
#include "utils/registry.cuh"

namespace {

__global__ void kernel(int M, int N, int K, float alpha, const float *A,
                       const float *B, float beta, float *C) {
  const uint x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint y = blockIdx.y * blockDim.y + threadIdx.y;
  if (y < M && x < N) {
    float tmp = 0.0f;
    for (int i = 0; i < K; ++i)
      tmp += A[y * K + i] * B[i * N + x];
    C[y * N + x] = alpha * tmp + beta * C[y * N + x];
  }
}

void launcher(int M, int N, int K, float alpha, const float *A,
              const float *B, float beta, float *C) {
  dim3 gridDim(CEIL_DIV(N, 32), CEIL_DIV(M, 32), 1);
  dim3 blockDim(32, 32, 1);
  kernel<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

}

REGISTER_KERNEL("gmem_coalesce");
