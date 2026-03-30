# Benchmarking of Eigen library usage


## Resize of Matrix or vector

### MacOS (m1 pro)

```text
Unable to determine clock rate from sysctl: hw.cpufrequency: No such file or directory
This does not affect benchmark measurements, only the metadata output.
***WARNING*** Failed to set thread affinity. Estimated CPU frequency may be incorrect.
2026-03-31T01:12:15+02:00
Running ./main
Run on (8 X 24 MHz CPU s)
CPU Caches:
  L1 Data 64 KiB
  L1 Instruction 128 KiB
  L2 Unified 4096 KiB (x8)
Load Average: 2.14, 2.70, 2.92
------------------------------------------------------------------------------------------------
Benchmark                                                      Time             CPU   Iterations
------------------------------------------------------------------------------------------------
Using dense matrix resize/iterations:100000/threads:1      46686 ns        46663 ns       100000
Using vector/iterations:100000/threads:1                   47075 ns        47060 ns       100000
```
