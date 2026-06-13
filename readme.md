## sgemm-cublas-bench

reimplementing https://siboehm.com/articles/22/CUDA-MMM w/o the code reference

Measured on an NVIDIA A40 (sm_86), 4096×4096×4096, 3 warmup + 10 timed runs.

| Kernel        | Time (ms) | TFLOP/s | % of cuBLAS | Slowdown | Correct |
|---------------|----------:|--------:|------------:|---------:|---------|
| cuBLAS        |     6.097 |   22.54 |      100.0% |     1.0x |     ref |
| vector_access |    11.850 |   11.60 |       51.4% |     1.9x |      OK |
| 2d_tile       |    13.064 |   10.52 |       46.7% |     2.1x |      OK |
| 1d_tile       |    35.729 |    3.85 |       17.1% |     5.9x |      OK |
| smem_chunk    |    68.567 |    2.00 |        8.9% |    11.2x |      OK |
| gmem_coalesce |    76.185 |    1.80 |        8.0% |    12.5x |      OK |
| naive         |   250.727 |    0.55 |        2.4% |    41.1x |      OK |
