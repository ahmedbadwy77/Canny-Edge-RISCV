# RISC-V Canny Edge Detection Pipeline 🚀

An optimized Canny Edge Detection system designed for the **RISC-V (RV64GCV)** architecture. This project uses **Vector Intrinsics (RVV)** to accelerate image processing stages.

## 🛠 Features
- **Full 7-Stage Pipeline:** Gaussian Blur, Sobel, Magnitude, Direction, NMS, Thresholding, and Hysteresis.
- **Hardware Acceleration:** RVV-optimized Magnitude and 5x5 Gaussian Blur.
- **TDD Approach:** 33 unit tests using **GoogleTest** to ensure 100% mathematical accuracy.
- **Performance Profiling:** Detailed benchmarking across different optimization levels (`-O0` to `-O3`).

## 📊 Performance Status (Phase 6)
| Stage | Scalar (C++) | RVV Optimized | Status |
| :--- | :---: | :---: | :---: |
| Gaussian Blur (5x5) | ✅ | ✅ | 🟢 Optimized |
| Sobel Filter | ✅ | 🔄 | In Progress |
| Magnitude | ✅ | ✅ | 🟢 Optimized |
| NMS | ✅ | ❌ | Target |

## 🚀 Getting Started
### Prerequisites
- RISC-V GNU Toolchain (rv64gcv)
- QEMU Emulator

### Build & Run
```bash
# Run Unit Tests
make test

# Run Profiling on QEMU
make run OPT=-O3

## Phase 5: Full Pipeline Profiling (7-Stage Breakdown)
Measured on QEMU using `-O3` optimization for a 256x256 image.

| Stage              | Time (ms)   | Percentage (%) | Status              |
|--------------------|-------------|----------------|---------------------|
| **Gaussian Blur** | 1127.91 ms  | 49.42%         | ✅ RVV Optimized     |
| Sobel Gradients    | 44.52 ms    | 1.95%          | 🔄 Scalar Baseline  |
| **Magnitude** | 199.06 ms   | 8.72%          | ✅ RVV Optimized     |
| Direction          | 102.85 ms   | 4.51%          | 🔄 Scalar Baseline  |
| NMS                | 646.78 ms   | 28.34%         | 🔄 Scalar Baseline  |
| Thresholding       | 98.21 ms    | 4.30%          | 🔄 Scalar Baseline  |
| Hysteresis         | 62.83 ms    | 2.75%          | 🔄 Scalar Baseline  |
| **Total Execution**| **2282.16 ms** | **100%** |                     |

> **Note:** Gaussian Blur shows higher absolute time due to QEMU's emulation of widening instructions, but it processes vectors in parallel.