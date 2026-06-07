#pragma once
#include "common.cuh"

class ScopedGpuTimer {
public:
  explicit ScopedGpuTimer(float &out_ms) : out_ms_(out_ms) {
    CUDA_CHECK(cudaEventCreate(&start_));
    CUDA_CHECK(cudaEventCreate(&stop_));
    CUDA_CHECK(cudaEventRecord(start_));
  }
  ~ScopedGpuTimer() {
    cudaEventRecord(stop_);
    cudaEventSynchronize(stop_);
    cudaEventElapsedTime(&out_ms_, start_, stop_);
    cudaEventDestroy(start_);
    cudaEventDestroy(stop_);
  }

private:
  cudaEvent_t start_, stop_;
  float &out_ms_;
};
