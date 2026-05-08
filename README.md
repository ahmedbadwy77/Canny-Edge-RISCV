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