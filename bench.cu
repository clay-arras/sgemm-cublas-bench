#include "utils/common.cuh"
#include "utils/registry.cuh"
#include "utils/timer.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

static double tflops(int M, int N, int K, double ms) {
  return 2.0 * M * N * K / (ms * 1e-3) / 1e12;
}

static int count_mismatches(int M, int N, const float *ref, const float *got,
                            float tol = 1e-3f) {
  int errors = 0;
  for (int i = 0; i < M * N; ++i) {
    float diff = fabsf(ref[i] - got[i]);
    float scale = fmaxf(fabsf(ref[i]), 1e-5f);
    if (diff / scale > tol)
      ++errors;
  }
  return errors;
}

struct Result {
  std::string name;
  double ms;
  double tflops;
  bool is_baseline;
  int errors;
};

int main() {
  const int M = 4096, N = 4096, K = 4096;
  const float alpha = 1.0f, beta = 0.0f;
  const int WARMUP = 3, RUNS = 10;
  const char *kBaseline = "cuBLAS";

  const size_t bytes_A = (size_t)M * K * sizeof(float);
  const size_t bytes_B = (size_t)K * N * sizeof(float);
  const size_t bytes_C = (size_t)M * N * sizeof(float);

  float *hA = (float *)malloc(bytes_A);
  float *hB = (float *)malloc(bytes_B);
  float *hC = (float *)malloc(bytes_C);
  float *hRef = (float *)malloc(bytes_C);

  srand(42);
  for (int i = 0; i < M * K; ++i)
    hA[i] = (float)rand() / RAND_MAX - 0.5f;
  for (int i = 0; i < K * N; ++i)
    hB[i] = (float)rand() / RAND_MAX - 0.5f;

  float *dA, *dB, *dC;
  CUDA_CHECK(cudaMalloc(&dA, bytes_A));
  CUDA_CHECK(cudaMalloc(&dB, bytes_B));
  CUDA_CHECK(cudaMalloc(&dC, bytes_C));
  CUDA_CHECK(cudaMemcpy(dA, hA, bytes_A, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB, hB, bytes_B, cudaMemcpyHostToDevice));

  const auto &entries = KernelRegistry::instance().entries();

  printf("Matrix: %dx%d  K=%d   work=%.2f TFLOP   (%d warmup, %d timed runs)\n",
         M, N, K, 2.0 * M * N * K / 1e12, WARMUP, RUNS);
  printf("Device: ");
  {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("%s   kernels registered: %zu\n\n", prop.name, entries.size());
  }

  const KernelEntry *baseline = nullptr;
  for (const auto &e : entries)
    if (e.name == kBaseline)
      baseline = &e;

  bool have_ref = false;
  if (baseline) {
    CUDA_CHECK(cudaMemset(dC, 0, bytes_C));
    baseline->fn(M, N, K, alpha, dA, dB, beta, dC);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(hRef, dC, bytes_C, cudaMemcpyDeviceToHost));
    have_ref = true;
  }

  std::vector<Result> results;
  for (const auto &e : entries) {
    CUDA_CHECK(cudaMemset(dC, 0, bytes_C));
    for (int i = 0; i < WARMUP; ++i)
      e.fn(M, N, K, alpha, dA, dB, beta, dC);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    float ms = 0.f;
    {
      ScopedGpuTimer t(ms);
      for (int i = 0; i < RUNS; ++i)
        e.fn(M, N, K, alpha, dA, dB, beta, dC);
    }
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(hC, dC, bytes_C, cudaMemcpyDeviceToHost));
    const int errors = have_ref ? count_mismatches(M, N, hRef, hC) : 0;
    const double per_run = ms / RUNS;
    results.push_back({e.name, per_run, tflops(M, N, K, per_run),
                       e.name == kBaseline, errors});
  }

  std::sort(results.begin(), results.end(),
            [](const Result &a, const Result &b) { return a.tflops > b.tflops; });

  double base_tflops = 0.0;
  for (const auto &r : results)
    if (r.is_baseline)
      base_tflops = r.tflops;

  printf("%-14s %12s %11s %14s %11s %9s\n", "Kernel", "Time (ms)", "TFLOP/s",
         "% of cuBLAS", "Slowdown", "Correct");
  printf("----------------------------------------------------------------------------\n");
  for (const auto &r : results) {
    char pct[32], slow[32];
    if (base_tflops > 0.0) {
      snprintf(pct, sizeof(pct), "%.1f%%", 100.0 * r.tflops / base_tflops);
      snprintf(slow, sizeof(slow), "%.1fx", base_tflops / r.tflops);
    } else {
      snprintf(pct, sizeof(pct), "-");
      snprintf(slow, sizeof(slow), "-");
    }
    const char *correct =
        r.is_baseline ? "ref" : (r.errors == 0 ? "OK" : "FAIL");
    printf("%-14s %12.3f %11.2f %14s %11s %9s\n", r.name.c_str(), r.ms,
           r.tflops, pct, slow, correct);
  }

  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);
  free(hA);
  free(hB);
  free(hC);
  free(hRef);
  return 0;
}
