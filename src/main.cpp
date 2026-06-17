#include <iostream>
#include <cstdlib>
#include <iomanip>
#include <chrono>
#include <cerrno>
#include <climits>
#include <vector>
#include "../include/image_io.h"
#include "../include/gaussian.h"
#include "../include/sobel.h"
#include "../include/magnitude.h"
#include "../include/direction.h"
#include "../include/nms.h"
#include "../include/threshold.h"
#include "../include/hysteresis.h"

static bool parse_positive_int(const char* text, int& value) {
    errno = 0;
    char* end = nullptr;
    long parsed = std::strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || parsed <= 0 || parsed > INT_MAX) {
        return false;
    }
    value = static_cast<int>(parsed);
    return true;
}

int main(int argc, char** argv) {
    int width = 256;
    int height = 256;
    int iterations = 100;

    if (argc >= 3) {
        if (!parse_positive_int(argv[1], width) || !parse_positive_int(argv[2], height)) {
            std::cerr << "Usage: " << argv[0] << " [width height [iterations]]\n";
            return 1;
        }
    }
    if (argc >= 4) {
        if (!parse_positive_int(argv[3], iterations)) {
            std::cerr << "Iterations must be a positive integer.\n";
            return 1;
        }
    }
    if (argc == 2 || argc > 4) {
        std::cerr << "Usage: " << argv[0] << " [width height [iterations]]\n";
        return 1;
    }
    if (width < 5 || height < 5 || width > INT_MAX / height) {
        std::cerr << "Image dimensions must be at least 5x5 and must not overflow.\n";
        return 1;
    }

    size_t size = static_cast<size_t>(width) * static_cast<size_t>(height);

    uint8_t* input_buf = read_raw_image("input.raw", width, height);
    if (!input_buf) {
        std::cout << "Warning: Input unavailable. Generating image in memory to continue profiling...\n";
        input_buf = static_cast<uint8_t*>(std::malloc(size));
        if (!input_buf) {
            std::cerr << "Error: Could not allocate input buffer.\n";
            return 1;
        }
        for (size_t i = 0; i < size; ++i) {
            input_buf[i] = i % 256;
        }
    }

    std::vector<uint8_t> blurred_data(size);
    std::vector<int16_t> grad_x(size);
    std::vector<int16_t> grad_y(size);
    std::vector<uint8_t> magnitude_data(size);
    std::vector<uint8_t> direction_data(size);
    std::vector<uint8_t> nms_data(size);
    std::vector<uint8_t> threshold_data(size);

    double t_gauss = 0, t_sobel = 0, t_mag = 0, t_dir = 0, t_nms = 0, t_thresh = 0, t_hyst = 0;

    for (int it = 0; it < iterations; ++it) {
        auto s = std::chrono::steady_clock::now();
#if defined(__riscv_vector)
        gaussian_blur_rvv(input_buf, blurred_data.data(), width, height);
#else
        gaussian_blur_scalar(input_buf, blurred_data.data(), width, height);
#endif
        auto e = std::chrono::steady_clock::now();
        t_gauss += std::chrono::duration<double, std::milli>(e - s).count();

        s = std::chrono::steady_clock::now();
#if defined(__riscv_vector)
        sobel_gradients_rvv(blurred_data.data(), grad_x.data(), grad_y.data(), width, height);
#else
        sobel_gradients_scalar(blurred_data.data(), grad_x.data(), grad_y.data(), width, height);
#endif
        e = std::chrono::steady_clock::now();
        t_sobel += std::chrono::duration<double, std::milli>(e - s).count();

        s = std::chrono::steady_clock::now();
#if defined(__riscv_vector)
        gradient_magnitude_rvv(grad_x.data(), grad_y.data(), magnitude_data.data(), width, height);
#else
        gradient_magnitude_scalar(grad_x.data(), grad_y.data(), magnitude_data.data(), width, height, MagMethod::L1);
#endif
        e = std::chrono::steady_clock::now();
        t_mag += std::chrono::duration<double, std::milli>(e - s).count();

        s = std::chrono::steady_clock::now();
#if defined(__riscv_vector)
        gradient_direction_rvv(grad_x.data(), grad_y.data(), direction_data.data(), width, height);
#else
        gradient_direction_scalar(grad_x.data(), grad_y.data(), direction_data.data(), width, height);
#endif
        e = std::chrono::steady_clock::now();
        t_dir += std::chrono::duration<double, std::milli>(e - s).count();

        s = std::chrono::steady_clock::now();
        non_max_suppression(magnitude_data.data(), direction_data.data(), nms_data.data(), width, height);
        e = std::chrono::steady_clock::now();
        t_nms += std::chrono::duration<double, std::milli>(e - s).count();

        s = std::chrono::steady_clock::now();
        uint8_t low, high;
        auto_threshold(nms_data.data(), size, low, high);
        double_threshold(nms_data.data(), threshold_data.data(), width, height, low, high);
        e = std::chrono::steady_clock::now();
        t_thresh += std::chrono::duration<double, std::milli>(e - s).count();

        s = std::chrono::steady_clock::now();
        hysteresis_tracing(threshold_data.data(), width, height);
        e = std::chrono::steady_clock::now();
        t_hyst += std::chrono::duration<double, std::milli>(e - s).count();
    }

    double total = t_gauss + t_sobel + t_mag + t_dir + t_nms + t_thresh + t_hyst;
    std::cout << "\n--- Phase 6: Profiling ---\n";
#if defined(__riscv_vector)
    std::cout << "Mode:            RISC-V RVV\n";
#elif defined(__riscv)
    std::cout << "Mode:            RISC-V Scalar\n";
#else
    std::cout << "Mode:            Host Scalar\n";
#endif
    std::cout << "Size:            " << width << "x" << height << "\n";
    std::cout << "Iterations:      " << iterations << "\n";
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Gaussian Blur:   " << (t_gauss/total)*100 << "% (" << t_gauss << " ms)\n";
    std::cout << "Sobel Gradients: " << (t_sobel/total)*100 << "% (" << t_sobel << " ms)\n";
    std::cout << "Magnitude:       " << (t_mag/total)*100 << "% (" << t_mag << " ms)\n";
    std::cout << "Direction:       " << (t_dir/total)*100 << "% (" << t_dir << " ms)\n";
    std::cout << "NMS:             " << (t_nms/total)*100 << "% (" << t_nms << " ms)\n";
    std::cout << "Thresholding:    " << (t_thresh/total)*100 << "% (" << t_thresh << " ms)\n";
    std::cout << "Hysteresis:      " << (t_hyst/total)*100 << "% (" << t_hyst << " ms)\n";
    std::cout << "Total Time:      " << total << " ms\n";

    if (!write_raw_image("output.raw", threshold_data.data(), width, height)) {
        std::free(input_buf);
        return 1;
    }

    std::free(input_buf);

    return 0;
}
