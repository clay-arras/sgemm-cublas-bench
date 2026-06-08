#include <cublas_v2.h>

#include "utils/common.cuh"
#include "utils/registry.cuh"

namespace {

cublasHandle_t handle() {
  static cublasHandle_t h = [] {
    cublasHandle_t tmp;
    CUBLAS_CHECK(cublasCreate(&tmp));
    return tmp;
  }();
  return h;
}

void launcher(int M, int N, int K, float alpha, const float *A,
              const float *B, float beta, float *C) {
  CUBLAS_CHECK(cublasSgemm(handle(), CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                           &alpha, B, N, A, K, &beta, C, N));
}

}

REGISTER_KERNEL("cuBLAS");
