# RISC-V Canny Edge Detection Pipeline 🚀

![C++](https://img.shields.io/badge/language-C%2B%2B-blue.svg)
![RISC-V](https://img.shields.io/badge/Architecture-RISC--V-orange.svg)
![GoogleTest](https://img.shields.io/badge/Testing-GoogleTest-green.svg)

An optimized Canny Edge Detection pipeline designed for the **RISC-V (RV64GCV)** architecture. This project leverages **RISC-V Vector Extension (RVV)** to accelerate high-latency image processing stages.

---

## 🛠 Features
- **Full 7-Stage Pipeline:** Gaussian Blur, Sobel, Magnitude, Direction, NMS, Thresholding, and Hysteresis.
- **Hardware Acceleration:** RVV-optimized kernels for Magnitude and 5x5 Gaussian Blur.
- **TDD Methodology:** 33 unit tests using **GoogleTest** to ensure 100% mathematical accuracy.
- **Performance Profiling:** Detailed benchmarking across optimization levels (`-O0` to `-O3`).

---

## 📂 Repository Structure

```text
.
├── .vscode/                  # VS Code configuration files
├── images/                   # Input images & sample data
│   └── test_image.raw
├── include/                  # Header files for pipeline stages
│   ├── direction.h
│   ├── gaussian.h
│   ├── hysteresis.h
│   ├── image_io.h
│   ├── magnitude.h
│   ├── nms.h
│   ├── sobel.h
│   └── threshold.h
├── src/                      # Source files & main pipeline
│   ├── image_io.cpp
│   └── main.cpp
├── tests/                    # GoogleTest unit tests
│   └── test_pipeline.cpp
├── Makefile                  # Build system for Host & RISC-V
├── RISCV_Canny_Documentation_v3.pdf
├── .gitignore
└── README.md
```

---

## 📊 Phase 5: Full Pipeline Profiling (Benchmark)

*Measured on QEMU using `-O3` optimization for a 256x256 image.*

| Pipeline Stage      | Execution Time (ms) | Time Percentage (%) | Status              |
|---------------------|---------------------|---------------------|---------------------|
| **Gaussian Blur**   | 1127.91 ms          | 49.42%              | ✅ RVV Optimized     |
| Sobel Gradients     | 44.52 ms            | 1.95%               | 🔄 Scalar Baseline  |
| **Magnitude**       | 199.06 ms           | 8.72%               | ✅ RVV Optimized     |
| Direction           | 102.85 ms           | 4.51%               | 🔄 Scalar Baseline  |
| NMS                 | 646.78 ms           | 28.34%              | 🔄 Scalar Baseline  |
| Thresholding        | 98.21 ms            | 4.30%               | 🔄 Scalar Baseline  |
| Hysteresis          | 62.83 ms            | 2.75%               | 🔄 Scalar Baseline  |
| **Total Execution** | **2282.16 ms**      | **100%**            |                     |

> **Note:** Gaussian Blur execution time is higher in QEMU due to software emulation of vector widening instructions (`vwmul`), but offers massive parallelism on native hardware.

---

## 🚀 Getting Started

### Prerequisites
- RISC-V GNU Toolchain (`rv64gcv`)
- QEMU Emulator

### Build & Run Instructions

```bash
# 1. Run Unit Tests (Native Host)
make test

# 2. Run Profiling (RISC-V Emulation)
make run OPT=-O3
```