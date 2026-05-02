#include <iostream>
#include <cstdlib>
#include <ctime>   // <-- added for timing
#include "../include/image_io.h"
#include "../include/gaussian.h"
#include "../include/sobel.h"
#include "../include/magnitude.h"
#include "../include/direction.h"

int main() {
    int width = 256;
    int height = 256;
    const char* filename = "images/test_image.raw";
    size_t size = width * height;

    uint8_t* dummy_data = (uint8_t*)aligned_alloc(32, size);
    if (!dummy_data) {
        std::cerr << "Error: Memory allocation failed!" << std::endl;
        return 1;
    }
    for (size_t i = 0; i < size; ++i) {
        dummy_data[i] = i % 256;
    }

    uint8_t* blurred_data = (uint8_t*)aligned_alloc(32, size);
    int16_t* grad_x = (int16_t*)aligned_alloc(32, size * sizeof(int16_t));
    int16_t* grad_y = (int16_t*)aligned_alloc(32, size * sizeof(int16_t));
    uint8_t* magnitude_data = (uint8_t*)aligned_alloc(32, size);
    uint8_t* direction_data = (uint8_t*)aligned_alloc(32, size);

    if (!blurred_data || !grad_x || !grad_y || !magnitude_data || !direction_data) {
        std::cerr << "Error: Memory allocation failed!" << std::endl;
        return 1;
    }

    // ---------------- TIMING HARNESS ----------------
    timespec start, end;
    int iterations = 100;
    clock_gettime(CLOCK_MONOTONIC, &start);

    for (int it = 0; it < iterations; ++it) {
        gaussian_blur_scalar(dummy_data, blurred_data, width, height);
        sobel_gradients_scalar(blurred_data, grad_x, grad_y, width, height);
        gradient_magnitude_scalar(grad_x, grad_y, magnitude_data, width, height, MagMethod::L1);
        gradient_direction_scalar(grad_x, grad_y, direction_data, width, height);
    }

    clock_gettime(CLOCK_MONOTONIC, &end);
    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    std::cout << "Elapsed time for " << iterations << " iterations: " << elapsed << " seconds" << std::endl;
    // ------------------------------------------------

#ifndef __riscv
    std::cout << "--- Running on HOST: Testing File I/O ---" << std::endl;
    write_raw_image(filename, dummy_data, width, height);
    std::cout << "Successfully saved dummy image." << std::endl;
    uint8_t* read_data = read_raw_image(filename, width, height);
    if (read_data) {
        std::cout << "Successfully read image back." << std::endl;
        free(read_data);
    }
#else
    std::cout << "--- Running on RISC-V (QEMU): Skipping File I/O ---" << std::endl;
    std::cout << "Memory allocation and setup successful!" << std::endl;
#endif

    free(dummy_data);
    free(blurred_data);
    free(grad_x);
    free(grad_y);
    free(magnitude_data);
    free(direction_data);

    std::cout << "Phase 4 completed successfully!" << std::endl;
    return 0;
}

