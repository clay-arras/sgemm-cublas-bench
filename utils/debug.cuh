#pragma once
#include <cmath>
#include <cstdio>

#ifndef DBG_WIDTH
#define DBG_WIDTH 9
#endif
#ifndef DBG_PREC
#define DBG_PREC 3
#endif
#ifndef DBG_MAX
#define DBG_MAX 16
#endif
#ifndef DBG_COLOR
#define DBG_COLOR 1
#endif

#define DBG_STR2(x) #x
#define DBG_STR(x) DBG_STR2(x)
#define DBG_FMT_F "%" DBG_STR(DBG_WIDTH) "." DBG_STR(DBG_PREC) "f"
#define DBG_FMT_S "%" DBG_STR(DBG_WIDTH) "s"

#if DBG_COLOR
#define DBG_C_RED "\033[31m"
#define DBG_C_YEL "\033[33m"
#define DBG_C_RST "\033[0m"
#else
#define DBG_C_RED ""
#define DBG_C_YEL ""
#define DBG_C_RST ""
#endif

namespace dbg {

__device__ inline int min_i(int a, int b) { return a < b ? a : b; }
__device__ inline bool finite_f(float v) { return isfinite(v); }

__device__ inline void print_mat(const char *label, const char *file, int line,
                                 const float *m, int rows, int cols, int ld,
                                 int maxr, int maxc) {
  int pr = min_i(rows, maxr), pc = min_i(cols, maxc);
  printf("== %s  (%dx%d, showing %dx%d) [dev] %s:%d ==\n", label, rows, cols, pr,
         pc, file, line);
  for (int i = 0; i < pr; ++i) {
    for (int j = 0; j < pc; ++j) {
      float v = m[(size_t)i * ld + j];
      if (finite_f(v))
        printf(DBG_FMT_F, v);
      else
        printf(DBG_C_YEL DBG_FMT_S DBG_C_RST, "nan/inf");
    }
    printf("%s\n", pc < cols ? "  ..." : "");
  }
  if (pr < rows)
    printf("  ...\n");
}

}

#ifndef DBG_DISABLE
#define DBG_MAT(label, ptr, rows, cols)                                        \
  ::dbg::print_mat((label), __FILE__, __LINE__, (ptr), (rows), (cols), (cols), \
                   DBG_MAX, DBG_MAX)
#define DBG_MAT_LD(label, ptr, rows, cols, ld)                                 \
  ::dbg::print_mat((label), __FILE__, __LINE__, (ptr), (rows), (cols), (ld),   \
                   DBG_MAX, DBG_MAX)
#define DBG_CORNER(label, ptr, rows, cols, n)                                  \
  ::dbg::print_mat((label), __FILE__, __LINE__, (ptr), (rows), (cols), (cols), \
                   (n), (n))
#else
#define DBG_MAT(label, ptr, rows, cols) ((void)0)
#define DBG_MAT_LD(label, ptr, rows, cols, ld) ((void)0)
#define DBG_CORNER(label, ptr, rows, cols, n) ((void)0)
#endif
