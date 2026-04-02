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

### Fedora (AMD server)

```text
2026-03-31T01:18:37+02:00
Running ./main
Run on (96 X 4000 MHz CPU s)
CPU Caches:
  L1 Data 32 KiB (x48)
  L1 Instruction 32 KiB (x48)
  L2 Unified 1024 KiB (x48)
  L3 Unified 36608 KiB (x2)
Load Average: 0.56, 0.45, 0.38
***WARNING*** CPU scaling is enabled, the benchmark real time measurements may be noisy and will incur extra overhead.
------------------------------------------------------------------------------------------------
Benchmark                                                      Time             CPU   Iterations
------------------------------------------------------------------------------------------------
Using dense matrix resize/iterations:100000/threads:1      69572 ns        69559 ns       100000
Using vector/iterations:100000/threads:1                   77340 ns        77327 ns       100000
```

```text
2026-03-31T01:26:02+02:00
Running ./main
Run on (96 X 4000 MHz CPU s)
CPU Caches:
  L1 Data 32 KiB (x48)
  L1 Instruction 32 KiB (x48)
  L2 Unified 1024 KiB (x48)
  L3 Unified 36608 KiB (x2)
Load Average: 0.67, 0.53, 0.42
***WARNING*** CPU scaling is enabled, the benchmark real time measurements may be noisy and will incur extra overhead.
-------------------------------------------------------------------------------------------------------------
Benchmark                                                                   Time             CPU   Iterations
-------------------------------------------------------------------------------------------------------------
Using vector/iterations:100000/threads:1                                77211 ns        77171 ns       100000
Using dense matrix resize/iterations:100000/threads:1                   69355 ns        69318 ns       100000
Using dense matrix conservative resize/iterations:100000/threads:1      71817 ns        71777 ns       100000
```
