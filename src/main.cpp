#include <iostream>
#include <cstdlib>
#include <iomanip>
#include <chrono> 
#include "../include/image_io.h"
#include "../include/gaussian.h"
#include "../include/sobel.h"
#include "../include/magnitude.h"
#include "../include/direction.h"

int main() {
    int width = 256;
    int height = 256;
    size_t size = width * height;

    // --- Memory Allocation ---
    uint8_t* dummy_data = (uint8_t*)aligned_alloc(32, size);
    for (size_t i = 0; i < size; ++i) {
        dummy_data[i] = i % 256;
    }

    uint8_t* blurred_data = (uint8_t*)aligned_alloc(32, size);
    int16_t* grad_x = (int16_t*)aligned_alloc(32, size * sizeof(int16_t));
    int16_t* grad_y = (int16_t*)aligned_alloc(32, size * sizeof(int16_t));
    uint8_t* magnitude_data = (uint8_t*)aligned_alloc(32, size);
    uint8_t* direction_data = (uint8_t*)aligned_alloc(32, size);

    // --- PHASE 5: PROFILING (MODERN C++ CHRONO) ---
    double time_gaussian = 0.0, time_sobel = 0.0, time_magnitude = 0.0, time_direction = 0.0;
    int iterations = 100;

    for (int it = 0; it < iterations; ++it) {
        // 1. Measure Gaussian Blur
        auto start = std::chrono::high_resolution_clock::now();
        gaussian_blur_scalar(dummy_data, blurred_data, width, height);
        auto end = std::chrono::high_resolution_clock::now();
        time_gaussian += std::chrono::duration<double, std::milli>(end - start).count();

        // 2. Measure Sobel Gradients
        start = std::chrono::high_resolution_clock::now();
        sobel_gradients_scalar(blurred_data, grad_x, grad_y, width, height);
        end = std::chrono::high_resolution_clock::now();
        time_sobel += std::chrono::duration<double, std::milli>(end - start).count();

        // 3. Measure Magnitude
        start = std::chrono::high_resolution_clock::now();
        gradient_magnitude_scalar(grad_x, grad_y, magnitude_data, width, height, MagMethod::L1);
        end = std::chrono::high_resolution_clock::now();
        time_magnitude += std::chrono::duration<double, std::milli>(end - start).count();

        // 4. Measure Direction
        start = std::chrono::high_resolution_clock::now();
        gradient_direction_scalar(grad_x, grad_y, direction_data, width, height);
        end = std::chrono::high_resolution_clock::now();
        time_direction += std::chrono::duration<double, std::milli>(end - start).count();
    }

    // --- Calculate Percentages ---
    double total_time = time_gaussian + time_sobel + time_magnitude + time_direction;

    std::cout << "\n--- Phase 5: Profiling Breakdown ---\n";
    std::cout << std::fixed << std::setprecision(2);
    
    if (total_time > 0.001) {
        std::cout << "Gaussian Blur:   " << (time_gaussian / total_time) * 100 << "% (" << time_gaussian << " ms)\n";
        std::cout << "Sobel Gradients: " << (time_sobel / total_time) * 100 << "% (" << time_sobel << " ms)\n";
        std::cout << "Magnitude:       " << (time_magnitude / total_time) * 100 << "% (" << time_magnitude << " ms)\n";
        std::cout << "Direction:       " << (time_direction / total_time) * 100 << "% (" << time_direction << " ms)\n";
    } else {
        std::cout << "Error: Execution too fast to measure accurately. Run more iterations.\n";
    }
    std::cout << "Total execution time: " << total_time << " ms\n";

    // --- Cleanup ---
    free(dummy_data);
    free(blurred_data);
    free(grad_x);
    free(grad_y);
    free(magnitude_data);
    free(direction_data);

    return 0;
}
