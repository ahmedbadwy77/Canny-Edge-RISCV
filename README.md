# RISC-V Canny Edge Detection Pipeline 🚀

An optimized Canny Edge Detection system designed for the **RISC-V (RV64GCV)** architecture. This project leverages **RISC-V Vector Extension (RVV)** to accelerate high-latency image processing stages.

---

## 🛠 Features
- [cite_start]**Full 7-Stage Pipeline:** Gaussian Blur, Sobel, Magnitude, Direction, NMS, Thresholding, and Hysteresis [cite: 9, 10-16].
- [cite_start]**Hardware Acceleration:** RVV-optimized kernels for Magnitude and 5x5 Gaussian Blur [cite: 41-42].
- [cite_start]**TDD Methodology:** 33 unit tests using **GoogleTest** to ensure 100% mathematical accuracy[cite: 5, 19, 29].
- [cite_start]**Performance Profiling:** Comprehensive benchmarking across optimization levels (`-O0` to `-O3`)[cite: 22, 38].

---

## 📊 Phase 5: Full Pipeline Profiling (Benchmark)
*Measured on QEMU using `-O3` optimization for a 256x256 image.*

| Stage | Time (ms) | Percentage (%) | Status |
| :--- | :---: | :---: | :--- |
| **Gaussian Blur** | 1127.91 ms | 49.42% | ✅ RVV Optimized |
| Sobel Gradients | 44.52 ms | 1.95% | 🔄 Scalar Baseline |
| **Magnitude** | 199.06 ms | 8.72% | ✅ RVV Optimized |
| Direction | 102.85 ms | 4.51% | 🔄 Scalar Baseline |
| NMS | 646.78 ms | 28.34% | 🔄 Scalar Baseline |
| Thresholding | 98.21 ms | 4.30% | 🔄 Scalar Baseline |
| Hysteresis | 62.83 ms | 2.75% | 🔄 Scalar Baseline |
| **Total Execution** | **2282.16 ms** | **100%** | |

> [cite_start]**Note:** Gaussian Blur execution time is higher in QEMU due to the software emulation of vector widening instructions, but offers massive parallelism on native hardware[cite: 25].

---

## 🚀 Getting Started

### Prerequisites
- [cite_start]RISC-V GNU Toolchain (`rv64gcv`) [cite: 4]
- [cite_start]QEMU Emulator [cite: 5]

### Build & Run Instructions
```bash
# 1. Run Unit Tests (Native Host)
make test

# 2. Run Profiling (RISC-V Emulation)
make run OPT=-O3