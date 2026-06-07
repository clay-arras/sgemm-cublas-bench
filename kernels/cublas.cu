#include <cublas_v2.h>

#include "utils/common.cuh"
#include "utils/registry.cuh"

static cublasHandle_t cublas_handle() {
  static cublasHandle_t handle = [] {
    cublasHandle_t h;
    CUBLAS_CHECK(cublasCreate(&h));
    return h;
  }();
  return handle;
}

void launch_cublas(int M, int N, int K, float alpha, const float *A,
                   const float *B, float beta, float *C) {
  CUBLAS_CHECK(cublasSgemm(cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                           &alpha, B, N, A, K, &beta, C, N));
}

REGISTER_KERNEL("cuBLAS", launch_cublas);
