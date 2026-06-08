#pragma once
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cassert>

#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))

#define DEVICE_ASSERT(cond) assert(cond)

#define CUDA_CHECK(call)                                                        \
  do {                                                                          \
    cudaError_t err_ = (call);                                                  \
    if (err_ != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at %s:%d - %s (%s)\n", __FILE__, __LINE__,    \
              cudaGetErrorString(err_), cudaGetErrorName(err_));                \
      exit(1);                                                                  \
    }                                                                           \
  } while (0)

#define CUDA_CHECK_KERNEL(label)                                                \
  do {                                                                          \
    cudaError_t launch_ = cudaPeekAtLastError();                                \
    if (launch_ != cudaSuccess) {                                               \
      fprintf(stderr, "[%s] launch error: %s (%s) at %s:%d\n", (label),         \
              cudaGetErrorString(launch_), cudaGetErrorName(launch_),           \
              __FILE__, __LINE__);                                              \
      exit(1);                                                                  \
    }                                                                           \
    cudaError_t exec_ = cudaDeviceSynchronize();                                \
    if (exec_ != cudaSuccess) {                                                 \
      fprintf(stderr, "[%s] execution error: %s (%s) at %s:%d\n", (label),      \
              cudaGetErrorString(exec_), cudaGetErrorName(exec_), __FILE__,     \
              __LINE__);                                                        \
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
