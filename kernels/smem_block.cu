#include "utils/common.cuh"
#include "utils/registry.cuh"

namespace {

// TODO: blocksize is probably a horrible horrible name for this
#define BLOCKSIZE 16

__global__ void kernel(int M, int N, int K, float alpha, const float *A,
                       const float *B, float beta, float *C) {

    int rowIdx = blockIdx.x * BLOCKSIZE;
    int colIdx = blockIdx.y * BLOCKSIZE;

    __shared__ float sA[BLOCKSIZE * BLOCKSIZE]; 
    __shared__ float sB[BLOCKSIZE * BLOCKSIZE]; 


    float tmp = 0.0;
    for (int slideIdx = 0; slideIdx < K; slideIdx += BLOCKSIZE) {
        int threadOffset = threadIdx.x + threadIdx.y * BLOCKSIZE;

        int Aidx_x = threadIdx.x + rowIdx;
        int Aidx_y = slideIdx + threadIdx.y;
        int Bidx_x = threadIdx.x + slideIdx;
        int Bidx_y = threadIdx.y + colIdx;

        if (Aidx_x < M && Aidx_y < K) {
            sA[threadOffset] = A[Aidx_x * K + Aidx_y];
        } else {
            sA[threadOffset] = 0.0;
        }
        if (Bidx_x < K && Bidx_y < N) {
            sB[threadOffset] = B[Bidx_x * N + Bidx_y];
        } else {
            sB[threadOffset] = 0.0;
        }
        
        __syncthreads();

        for (int i = 0; i < BLOCKSIZE; i++)
            tmp += sA[threadIdx.x + BLOCKSIZE * i] * sB[i + BLOCKSIZE * threadIdx.y];

        __syncthreads();
    }
    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < M && y < N)
        C[x * N + y] = alpha * tmp + beta * C[x * N + y];
}

void launcher(int M, int N, int K, float alpha, const float *A,
              const float *B, float beta, float *C) {
  dim3 gridDim(CEIL_DIV(M, BLOCKSIZE), CEIL_DIV(N, BLOCKSIZE), 1);
  dim3 blockDim(BLOCKSIZE, BLOCKSIZE, 1);
  kernel<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

}

REGISTER_KERNEL("smem_block");
