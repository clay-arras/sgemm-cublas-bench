#include "utils/check.cuh"
#include "utils/common.cuh"
#include "utils/registry.cuh"
#include "utils/timer.cuh"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

static double tflops(int M, int N, int K, double ms) {
  return 2.0 * M * N * K / (ms * 1e-3) / 1e12;
}

enum class MatInit { Ones, Rand, Seq };

static void fill_matrix(float *p, size_t n, MatInit mode) {
  switch (mode) {
  case MatInit::Ones:
    for (size_t i = 0; i < n; ++i)
      p[i] = 1.0f;
    break;
  case MatInit::Rand:
    for (size_t i = 0; i < n; ++i)
      p[i] = (float)rand() / RAND_MAX - 0.5f;
    break;
  case MatInit::Seq:
    for (size_t i = 0; i < n; ++i)
      p[i] = (float)i;
    break;
  }
}

static void usage(const char *prog) {
  fprintf(stderr,
          "usage: %s [--size=N] [--mat=ones|rand|seq]\n"
          "  --size=N   square matrix dimension (default 4096)\n"
          "  --mat=...  input fill: ones, rand (default), or seq\n",
          prog);
}

struct Result {
  std::string name;
  double ms;
  double tflops;
  bool is_baseline;
  Check check;
};

int main(int argc, char **argv) {
  int S = 4096;
  MatInit mat = MatInit::Rand;
  const char *mat_name = "rand";

  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a.rfind("--size=", 0) == 0) {
      S = atoi(a.c_str() + 7);
      if (S <= 0) {
        fprintf(stderr, "error: --size must be a positive integer\n");
        usage(argv[0]);
        return 1;
      }
    } else if (a.rfind("--mat=", 0) == 0) {
      std::string v = a.substr(6);
      if (v == "ones") {
        mat = MatInit::Ones;
        mat_name = "ones";
      } else if (v == "rand") {
        mat = MatInit::Rand;
        mat_name = "rand";
      } else if (v == "seq") {
        mat = MatInit::Seq;
        mat_name = "seq";
      } else {
        fprintf(stderr, "error: --mat must be one of ones, rand, seq\n");
        usage(argv[0]);
        return 1;
      }
    } else if (a == "--help" || a == "-h") {
      usage(argv[0]);
      return 0;
    } else {
      fprintf(stderr, "error: unknown argument '%s'\n", a.c_str());
      usage(argv[0]);
      return 1;
    }
  }

  const int M = S, N = S, K = S;
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
  fill_matrix(hA, (size_t)M * K, mat);
  fill_matrix(hB, (size_t)K * N, mat);

  float *dA, *dB, *dC;
  CUDA_CHECK(cudaMalloc(&dA, bytes_A));
  CUDA_CHECK(cudaMalloc(&dB, bytes_B));
  CUDA_CHECK(cudaMalloc(&dC, bytes_C));
  CUDA_CHECK(cudaMemcpy(dA, hA, bytes_A, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB, hB, bytes_B, cudaMemcpyHostToDevice));

  const auto &entries = KernelRegistry::instance().entries();

  printf("Matrix: %dx%d  K=%d  mat=%s   work=%.2f TFLOP   (%d warmup, %d timed "
         "runs)\n",
         M, N, K, mat_name, 2.0 * M * N * K / 1e12, WARMUP, RUNS);
  printf("Device: ");
  {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("%s   kernels registered: %zu\n\n", prop.name, entries.size());
  }

  CUDA_CHECK(cudaMemset(dC, 0, bytes_C));
  reference_sgemm(M, N, K, alpha, dA, dB, beta, dC);
  CUDA_CHECK_KERNEL("reference");
  CUDA_CHECK(cudaMemcpy(hRef, dC, bytes_C, cudaMemcpyDeviceToHost));

  std::vector<Result> results;
  for (const auto &e : entries) {
    CUDA_CHECK(cudaMemcpy(dA, hA, bytes_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB, bytes_B, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dC, 0, bytes_C));

    for (int i = 0; i < WARMUP; ++i)
      e.fn(M, N, K, alpha, dA, dB, beta, dC);
    CUDA_CHECK_KERNEL(e.name.c_str());

    float ms = 0.f;
    {
      ScopedGpuTimer t(ms);
      for (int i = 0; i < RUNS; ++i)
        e.fn(M, N, K, alpha, dA, dB, beta, dC);
    }
    CUDA_CHECK_KERNEL(e.name.c_str());

    CUDA_CHECK(cudaMemcpy(hC, dC, bytes_C, cudaMemcpyDeviceToHost));
    const Check chk = check_result(M, N, hRef, hC);
    const double per_run = ms / RUNS;
    results.push_back({e.name, per_run, tflops(M, N, K, per_run),
                       e.name == kBaseline, chk});
  }

  std::sort(results.begin(), results.end(),
            [](const Result &a, const Result &b) { return a.tflops > b.tflops; });

  double base_tflops = 0.0;
  for (const auto &r : results)
    if (r.is_baseline)
      base_tflops = r.tflops;

  printf("%-14s %12s %11s %14s %11s %11s %9s\n", "Kernel", "Time (ms)",
         "TFLOP/s", "% of cuBLAS", "Slowdown", "rel_fro", "Correct");
  printf("--------------------------------------------------------------------------------------\n");
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
        r.is_baseline
            ? "ref"
            : (!r.check.finite ? "NaN/Inf"
                               : (check_passed(r.check) ? "OK" : "FAIL"));
    printf("%-14s %12.3f %11.2f %14s %11s %11.2e %9s\n", r.name.c_str(), r.ms,
           r.tflops, pct, slow, r.check.rel_fro, correct);
    if (!r.is_baseline && !check_passed(r.check)) {
      const int idx = r.check.max_idx;
      printf("    \\_ worst element: max_rel=%.2e at (%d,%d)%s\n",
             r.check.max_rel, idx >= 0 ? idx / N : -1,
             idx >= 0 ? idx % N : -1,
             r.check.finite ? "" : "  [non-finite output present]");
    }
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
