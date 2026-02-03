# Benchmarking of Eigen library usage


## Dense vs Sparse

```text
Run on (8 X 24 MHz CPU s)
CPU Caches:
  L1 Data 64 KiB
  L1 Instruction 128 KiB
  L2 Unified 4096 KiB (x8)
Load Average: 3.59, 3.30, 3.21
-----------------------------------------------------------------------------------------------------------
Benchmark                                                                 Time             CPU   Iterations
-----------------------------------------------------------------------------------------------------------
Using sparse matrix/iterations:10/threads:1                          154858 ns       154200 ns           10
Using sparse matrix (only randomization)/iterations:10/threads:1     129362 ns       129400 ns           10
Using dense matrix/iterations:10/threads:1                        728878437 ns    728162800 ns           10
```
