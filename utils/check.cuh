#pragma once
#include "common.cuh"

#include <cublas_v2.h>

#include <cmath>

constexpr double kRelFroTol = 1e-4;

struct Check {
  double rel_fro;
  double max_rel;
  int max_idx;
  bool finite;
};

inline Check check_result(int M, int N, const float *ref, const float *got) {
  double num = 0, den = 0, max_rel = 0;
  int max_idx = -1;
  bool finite = true;
  for (int i = 0; i < M * N; ++i) {
    if (!std::isfinite(got[i]))
      finite = false;
    double d = (double)got[i] - (double)ref[i];
    num += d * d;
    den += (double)ref[i] * (double)ref[i];
    double scale = fmax(fabs((double)ref[i]), 1e-30);
    double rel = fabs(d) / scale;
    if (rel > max_rel) {
      max_rel = rel;
      max_idx = i;
    }
  }
  double rel_fro = den > 0 ? sqrt(num) / sqrt(den) : sqrt(num);
  return {rel_fro, max_rel, max_idx, finite};
}

inline void reference_sgemm(int M, int N, int K, float alpha, const float *dA,
                            const float *dB, float beta, float *dC) {
  static cublasHandle_t h = [] {
    cublasHandle_t t;
    CUBLAS_CHECK(cublasCreate(&t));
    CUBLAS_CHECK(cublasSetMathMode(t, CUBLAS_PEDANTIC_MATH));
    return t;
  }();
  CUBLAS_CHECK(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, dB, N,
                           dA, K, &beta, dC, N));
}

inline bool check_passed(const Check &c) {
  return c.finite && c.rel_fro < kRelFroTol;
}
