#pragma once
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))

#define CUDA_CHECK(call)                                                        \
  do {                                                                          \
    cudaError_t err_ = (call);                                                  \
    if (err_ != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__,         \
              cudaGetErrorString(err_));                                        \
      exit(1);                                                                  \
    }                                                                           \
  } while (0)

#define CUBLAS_CHECK(call)                                                      \
  do {                                                                          \
    cublasStatus_t st_ = (call);                                                \
    if (st_ != CUBLAS_STATUS_SUCCESS) {                                         \
      fprintf(stderr, "cuBLAS error at %s:%d - code %d\n", __FILE__, __LINE__,  \
              (int)st_);                                                        \
      exit(1);                                                                  \
    }                                                                           \
  } while (0)
