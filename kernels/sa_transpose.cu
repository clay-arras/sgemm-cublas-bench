#include "utils/common.cuh"
#include "utils/registry.cuh"
#include "utils/debug.cuh"

namespace {

#define T 8
#define CH_D 128
#define CH_K 16

__global__ void kernel(int M, int N, int K, float alpha, const float *A,
                       const float *B, float beta, float *C) {
    static_assert(CH_K * CH_D == CH_D * CH_D / T); 
    static_assert(T % 4 == 0);
    static_assert(CH_K % T == 0); // for sA trick

    __align__(16) __shared__ float sA[CH_D * CH_K];  
    __align__(16) __shared__ float sB[CH_D * CH_K]; 

    __align__(16) float tmp[T * T] = {0.0f};
    __align__(16) float sA_tmp[T] = {0.0f};
    __align__(16) float sB_tmp[T] = {0.0f};

    int threadIdx_x = threadIdx.x / CH_K;
    int threadIdx_y = threadIdx.x % CH_K;

    for (int slideIdx = 0; slideIdx < K; slideIdx += CH_K) { 
        /* int Aidx_x = threadIdx.x * T + blockIdx.x * CH_K * T; */ 
        /* int Aidx_y = slideIdx + threadIdx.y; */
        // TODO doesn't work yet

        int rthreadIdx_x = threadIdx.x / CH_D;
        int rthreadIdx_y = threadIdx.x % CH_D; 

        int Aidx_x = threadIdx.x * T + blockIdx.x * CH_K * T; 
        int Aidx_y = slideIdx + threadIdx.y;

        int Bidx_x = threadIdx_x + slideIdx; 
        int Bidx_y = threadIdx_y * T + blockIdx.y * CH_K * T;

#pragma unroll
        for (int i = 0; i < T; i++) {
            if (Aidx_x + i < M && Aidx_y < K) {
                sA[threadIdx.y * CH_D + (threadIdx.x * T + i)] = A[(Aidx_x + i) * K + Aidx_y];
            } else {
                sA[threadIdx.y * CH_D + (threadIdx.x * T + i)] = 0.0;
            }
        }

#pragma unroll
        for (int i = 0; i < T; i += 4) {
            int sB_base = threadIdx_x * CH_D + (threadIdx_y * T + i);
            if (Bidx_x < K && Bidx_y + i + 3 < N) {
                reinterpret_cast<float4 *>(&sB[sB_base])[0] =
                    reinterpret_cast<const float4 *>(&B[Bidx_x * N + (Bidx_y + i)])[0];
            } else {
                for (int k = 0; k < 4; k++) {
                    if (Bidx_x < K && Bidx_y + i + k < N)
                        sB[sB_base + k] = B[Bidx_x * N + (Bidx_y + i + k)];
                    else
                        sB[sB_base + k] = 0.0;
                }
            }
        }

        __syncthreads();

        for (int kIdx = 0; kIdx < CH_K; kIdx++) {
#pragma unroll
            for (int r = 0; r < T / 4; r++) {
                reinterpret_cast<float4*>(sA_tmp)[r] =
                    reinterpret_cast<const float4*>(&sA[kIdx * CH_D + (threadIdx_x * T)])[r];
                reinterpret_cast<float4*>(sB_tmp)[r] =
                    reinterpret_cast<const float4*>(&sB[kIdx * CH_D + (threadIdx_y * T)])[r];
            }
            
            for (int i = 0; i < T; i++) {
                for (int j = 0; j < T; j++) {
                    tmp[i * T + j] += sA_tmp[i] * sB_tmp[j];
                }
            }
        }
        __syncthreads();
    }

    for (int i = 0; i < T; i++) {
        int resIdx_x = blockIdx.x * blockDim.x * T + threadIdx_x * T + i;
        for (int j = 0; j < T; j += 4) {
            int resIdx_y = blockIdx.y * blockDim.y * T + threadIdx_y * T + j;

            if (resIdx_x < M && resIdx_y + 3 < N) {
                float *cptr = &C[resIdx_x * N + resIdx_y];
                float4 c = reinterpret_cast<float4 *>(cptr)[0];
                c.x = alpha * tmp[i * T + j + 0] + beta * c.x;
                c.y = alpha * tmp[i * T + j + 1] + beta * c.y;
                c.z = alpha * tmp[i * T + j + 2] + beta * c.z;
                c.w = alpha * tmp[i * T + j + 3] + beta * c.w;
                reinterpret_cast<float4 *>(cptr)[0] = c;
            } else {
                for (int k = 0; k < 4; k++) {
                    if (resIdx_x < M && resIdx_y + k < N)
                        C[resIdx_x * N + resIdx_y + k] =
                            alpha * tmp[i * T + j + k] + beta * C[resIdx_x * N + resIdx_y + k];
                }
            }
        }
    }
}

void launcher(int M, int N, int K, float alpha, const float *A,
              const float *B, float beta, float *C) {
  dim3 gridDim(CEIL_DIV(M, CH_K * T), CEIL_DIV(N, CH_K * T), 1);
  dim3 blockDim(CH_K * CH_K, 1);
  kernel<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

}

REGISTER_KERNEL("sa_transpose");
