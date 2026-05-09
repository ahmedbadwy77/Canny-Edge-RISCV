#include <iostream>
#include <cstdlib>
#include <iomanip>
#include <chrono>
#include "../include/image_io.h"
#include "../include/gaussian.h"
#include "../include/sobel.h"
#include "../include/magnitude.h"
#include "../include/direction.h"
#include "../include/nms.h"
#include "../include/threshold.h"
#include "../include/hysteresis.h"

int main() {
    int width = 256;
    int height = 256;
    size_t size = width * height;

    // --- 1. Attempt to load real image, fallback to memory generation if emulator blocks I/O ---
    uint8_t* input_buf = read_raw_image("input.raw", width, height);
    if (!input_buf) {
        std::cout << "Warning: Emulator blocked file I/O. Generating image in memory to continue profiling...\n";
        input_buf = (uint8_t*)aligned_alloc(32, size);
        for (size_t i = 0; i < size; ++i) {
            input_buf[i] = i % 256;
        }
    }

    // --- Allocate aligned memory for all pipeline stages ---
    uint8_t* blurred_data = (uint8_t*)aligned_alloc(32, size);
    int16_t* grad_x = (int16_t*)aligned_alloc(32, size * sizeof(int16_t));
    int16_t* grad_y = (int16_t*)aligned_alloc(32, size * sizeof(int16_t));
    uint8_t* magnitude_data = (uint8_t*)aligned_alloc(32, size);
    uint8_t* direction_data = (uint8_t*)aligned_alloc(32, size);
    uint8_t* nms_data = (uint8_t*)aligned_alloc(32, size);
    uint8_t* threshold_data = (uint8_t*)aligned_alloc(32, size);

    // --- Performance Profiling ---
    double t_gauss = 0, t_sobel = 0, t_mag = 0, t_dir = 0, t_nms = 0, t_thresh = 0, t_hyst = 0;
    int iterations = 10;

    for (int it = 0; it < iterations; ++it) {
        auto s = std::chrono::high_resolution_clock::now();
#ifdef __riscv
        gaussian_blur_rvv(input_buf, blurred_data, width, height);
#else
        gaussian_blur_scalar(input_buf, blurred_data, width, height);
#endif
        auto e = std::chrono::high_resolution_clock::now();
        t_gauss += std::chrono::duration<double, std::milli>(e - s).count();

        s = std::chrono::high_resolution_clock::now();
// --- THIS IS THE NEW PART FOR SOBEL RVV ---
#ifdef __riscv
        sobel_gradients_rvv(blurred_data, grad_x, grad_y, width, height);
#else
        sobel_gradients_scalar(blurred_data, grad_x, grad_y, width, height);
#endif
// ------------------------------------------
        e = std::chrono::high_resolution_clock::now();
        t_sobel += std::chrono::duration<double, std::milli>(e - s).count();

        s = std::chrono::high_resolution_clock::now();
#ifdef __riscv
        gradient_magnitude_rvv(grad_x, grad_y, magnitude_data, width, height);
#else
        gradient_magnitude_scalar(grad_x, grad_y, magnitude_data, width, height, MagMethod::L1);
#endif
        e = std::chrono::high_resolution_clock::now();
        t_mag += std::chrono::duration<double, std::milli>(e - s).count();

        s = std::chrono::high_resolution_clock::now();
        gradient_direction_scalar(grad_x, grad_y, direction_data, width, height);
        e = std::chrono::high_resolution_clock::now();
        t_dir += std::chrono::duration<double, std::milli>(e - s).count();

        s = std::chrono::high_resolution_clock::now();
        non_max_suppression(magnitude_data, direction_data, nms_data, width, height);
        e = std::chrono::high_resolution_clock::now();
        t_nms += std::chrono::duration<double, std::milli>(e - s).count();

        s = std::chrono::high_resolution_clock::now();
        uint8_t low, high;
        auto_threshold(nms_data, size, low, high);
        double_threshold(nms_data, threshold_data, width, height, low, high);
        e = std::chrono::high_resolution_clock::now();
        t_thresh += std::chrono::duration<double, std::milli>(e - s).count();

        s = std::chrono::high_resolution_clock::now();
        hysteresis_tracing(threshold_data, width, height);
        e = std::chrono::high_resolution_clock::now();
        t_hyst += std::chrono::duration<double, std::milli>(e - s).count();
    }

    double total = t_gauss + t_sobel + t_mag + t_dir + t_nms + t_thresh + t_hyst;
    std::cout << "\n--- Phase 6: Profiling (RVV Sobel Included) ---\n";
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Gaussian Blur:   " << (t_gauss/total)*100 << "% (" << t_gauss << " ms)\n";
    std::cout << "Sobel Gradients: " << (t_sobel/total)*100 << "% (" << t_sobel << " ms)\n";
    std::cout << "Magnitude:       " << (t_mag/total)*100 << "% (" << t_mag << " ms)\n";
    std::cout << "Direction:       " << (t_dir/total)*100 << "% (" << t_dir << " ms)\n";
    std::cout << "NMS:             " << (t_nms/total)*100 << "% (" << t_nms << " ms)\n";
    std::cout << "Thresholding:    " << (t_thresh/total)*100 << "% (" << t_thresh << " ms)\n";
    std::cout << "Hysteresis:      " << (t_hyst/total)*100 << "% (" << t_hyst << " ms)\n";
    std::cout << "Total Time:      " << total << " ms\n";

    free(input_buf); free(blurred_data); free(grad_x); free(grad_y);
    free(magnitude_data); free(direction_data); free(nms_data); free(threshold_data);
    
    return 0;
}
