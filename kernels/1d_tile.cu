#include "utils/common.cuh"
#include "utils/registry.cuh"
#include "utils/debug.cuh"

namespace {

#define T 8
#define CH_D 32
#define CH_K 4 

// TODO: swap axis cache locality optimization?
__global__ void kernel(int M, int N, int K, float alpha, const float *A,
                       const float *B, float beta, float *C) {
    __shared__ float sA[CH_D * CH_K]; 
    __shared__ float sB[CH_D * CH_K]; 

    assert(CH_K * CH_D == CH_D * CH_D / T); 

    float tmp[T] = {0.0};
    for (int slideIdx = 0; slideIdx < K; slideIdx += CH_K) { 
        int Aidx_x = threadIdx.y + blockIdx.x * CH_K * T; // fuckass weird rotation hack
        int Aidx_y = slideIdx + threadIdx.x;

        int Bidx_x = threadIdx.x + slideIdx;
        int Bidx_y = threadIdx.y + blockIdx.y * CH_D;

        if (Aidx_x < M && Aidx_y < K) {
            sA[threadIdx.x * CH_D + threadIdx.y] = A[Aidx_x * K + Aidx_y];
        } else {
            sA[threadIdx.x * CH_D + threadIdx.y] = 0.0;
        }
        if (Bidx_x < K && Bidx_y < N) {
            sB[threadIdx.x * CH_D + threadIdx.y] = B[Bidx_x * N + Bidx_y];
        } else {
            sB[threadIdx.x * CH_D + threadIdx.y] = 0.0;
        }
        
        __syncthreads();

        for (int i = 0; i < CH_K; i++) {
            float sB_tmp = sB[i * CH_D + threadIdx.y];
            
            for (int tmpIdx = 0; tmpIdx < T; tmpIdx++) {
                tmp[tmpIdx] += sA[i * CH_D + (threadIdx.x*T+tmpIdx)] * sB_tmp;
            }
        }

        __syncthreads();
    }

    for (int tmpIdx = 0; tmpIdx < T; tmpIdx++) {
        int resIdx_x = blockIdx.x * blockDim.x * T + threadIdx.x * T + tmpIdx;
        int resIdx_y = blockIdx.y * blockDim.y + threadIdx.y;
        if (resIdx_x < M && resIdx_y < N)
            C[resIdx_x * N + resIdx_y] = alpha * tmp[tmpIdx] + beta * C[resIdx_x * N + resIdx_y];
    }
}

void launcher(int M, int N, int K, float alpha, const float *A,
              const float *B, float beta, float *C) {
  dim3 gridDim(CEIL_DIV(M, CH_K), CEIL_DIV(N, CH_D), 1);
  dim3 blockDim(CH_K, CH_D, 1);
  kernel<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

}

REGISTER_KERNEL("1d_tile");
