## sgemm-cublas-bench

reimplementing https://siboehm.com/articles/22/CUDA-MMM w/o the code reference


| Kernel        | Time (ms) | TFLOP/s | % of cuBLAS | Slowdown | Correct |
|---------------|----------:|--------:|------------:|---------:|---------|
| cuBLAS        |     5.882 |   23.37 |      100.0% |     1.0x |     ref |
| smem_block    |    66.932 |    2.05 |        8.8% |    11.4x |      OK |
| gmem_coalesce |    85.083 |    1.62 |        6.9% |    14.5x |      OK |
| naive         |   491.261 |    0.28 |        1.2% |    83.5x |      OK |
