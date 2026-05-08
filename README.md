# Canny-Edge-RISCV
Embedded Project
## Phase 5: Profiling Breakdown (Optimization: -O3)
Current profiling results on QEMU after initial RVV implementation for Magnitude and Gaussian Blur.

| Stage | Time (ms) | Percentage (%) | Status |
| :--- | :--- | :--- | :--- |
| **Gaussian Blur** | **1127.91** | **49.42%** | **RVV Optimized** |
| Sobel Gradients | 44.52 | 1.95% | Scalar |
| **Magnitude** | **199.06** | **8.72%** | **RVV Optimized** |
| Direction | 102.85 | 4.51% | Scalar |
| NMS | 646.78 | 28.34% | Scalar |
| Thresholding | 98.21 | 4.30% | Scalar |
| Hysteresis | 62.83 | 2.75% | Scalar |
| **Total Time** | **2282.16 ms** | **100%** | |

> **Note:** Absolute time is higher due to QEMU software emulation of vector widening instructions.