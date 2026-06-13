#include "utils/common.cuh"
#include "utils/registry.cuh"
#include "utils/debug.cuh"

namespace {

#define T 8
#define CH_D 128
#define CH_K 16

__global__ void kernel(int M, int N, int K, float alpha, const float *A,
                       const float *B, float beta, float *C) {
    __shared__ float sA[CH_D * CH_K]; 
    __shared__ float sB[CH_D * CH_K]; 

    assert(CH_K * CH_D == CH_D * CH_D / T); 

    float tmp[T * T] = {0.0};
    float sA_tmp[T] = {0.0};
    float sB_tmp[T] = {0.0};
    for (int slideIdx = 0; slideIdx < K; slideIdx += CH_K) { 
        int Aidx_x = threadIdx.x * T + blockIdx.x * CH_K * T; 
        int Aidx_y = slideIdx + threadIdx.y;

        int Bidx_x = threadIdx.x + slideIdx; 
        int Bidx_y = threadIdx.y * T + blockIdx.y * CH_K * T;

        for (int i = 0; i < T; i++) {
            if (Aidx_x + i < M && Aidx_y < K) { // cursed indexing :p
                sA[threadIdx.y * CH_D + (threadIdx.x * T + i)] = A[(Aidx_x + i) * K + Aidx_y];
            } else {
                sA[threadIdx.y * CH_D + (threadIdx.x * T + i)] = 0.0;
            }
            if (Bidx_x < K && Bidx_y + i < N) {
                sB[threadIdx.x * CH_D + (threadIdx.y * T + i)] = B[Bidx_x * N + (Bidx_y + i)];
            } else {
                sB[threadIdx.x * CH_D + (threadIdx.y * T + i)] = 0.0;
            }
        }

        __syncthreads();

        for (int kIdx = 0; kIdx < CH_K; kIdx++) {
            for (int tmpIdx = 0; tmpIdx < T; tmpIdx++) {
                // cache locality efficient lol
                sA_tmp[tmpIdx] = sA[kIdx * CH_D + (threadIdx.x * T + tmpIdx)];
                sB_tmp[tmpIdx] = sB[kIdx * CH_D + (threadIdx.y * T + tmpIdx)];
            }
            
            for (int i = 0; i < T; i++) {
                for (int j = 0; j < T; j++) {
                    tmp[i * T + j] = tmp[i * T + j] + sA_tmp[i] * sB_tmp[j];
                }
            }
        }
        __syncthreads();
    }

    for (int i = 0; i < T; i++) {
        for (int j = 0; j < T; j++) {
            int resIdx_x = blockIdx.x * blockDim.x * T + threadIdx.x * T + i;
            int resIdx_y = blockIdx.y * blockDim.y * T + threadIdx.y * T + j;

            if (resIdx_x < M && resIdx_y < N)
                C[resIdx_x * N + resIdx_y] = alpha * tmp[i * T + j] + beta * C[resIdx_x * N + resIdx_y];
        }
    }
}

void launcher(int M, int N, int K, float alpha, const float *A,
              const float *B, float beta, float *C) {
  dim3 gridDim(CEIL_DIV(M, CH_K * T), CEIL_DIV(N, CH_K * T), 1);
  dim3 blockDim(CH_K, CH_K, 1);
  kernel<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

}

REGISTER_KERNEL("2d_tile");
