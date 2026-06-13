#include "utils/common.cuh"
#include "utils/registry.cuh"

namespace {

#define BLOCKSIZE 16

__global__ void kernel(int M, int N, int K, float alpha, const float *A,
                       const float *B, float beta, float *C) {
  const uint x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < M && y < N) {
    float tmp = 0.0f;
    for (int i = 0; i < K; ++i)
      tmp += A[x * K + i] * B[i * N + y];
    C[x * N + y] = alpha * tmp + beta * C[x * N + y];
  }
}

void launcher(int M, int N, int K, float alpha, const float *A,
              const float *B, float beta, float *C) {
  dim3 gridDim(CEIL_DIV(M, BLOCKSIZE), CEIL_DIV(N, BLOCKSIZE), 1);
  dim3 blockDim(BLOCKSIZE, BLOCKSIZE, 1);
  kernel<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

}

REGISTER_KERNEL("naive");
