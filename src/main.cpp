#include <iostream>
#include <cstdlib>
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



    // 2. حجز ميموري وتطبيق Gaussian Blur
    uint8_t* blurred_data = (uint8_t*)aligned_alloc(32, size);
    if (!blurred_data) {
        std::cerr << "Error: Memory allocation for Gaussian failed!" << std::endl;
        return 1;
    }
    
    gaussian_blur_scalar(dummy_data, blurred_data, width, height);
    std::cout << "Gaussian Blur applied successfully!" << std::endl;

  
    int16_t* grad_x = (int16_t*)aligned_alloc(32, size * sizeof(int16_t));
    int16_t* grad_y = (int16_t*)aligned_alloc(32, size * sizeof(int16_t));

    if (!grad_x || !grad_y) {
        std::cerr << "Error: Memory allocation for Sobel failed!" << std::endl;
        return 1;
    }

    sobel_gradients_scalar(blurred_data, grad_x, grad_y, width, height);
    std::cout << "Sobel Gradients (Gx, Gy) calculated successfully!" << std::endl;

// ---------------------------------------------------------
    // 3. Gradient Magnitude
    // ---------------------------------------------------------
    uint8_t* magnitude_data = (uint8_t*)aligned_alloc(32, size);
    if (!magnitude_data) {
        std::cerr << "Error: Memory allocation for Magnitude failed!" << std::endl;
        return 1;
    }
    gradient_magnitude_scalar(grad_x, grad_y, magnitude_data, width, height, MagMethod::L1);
    std::cout << "Gradient Magnitude calculated successfully!" << std::endl;

    // ---------------------------------------------------------
    // 4. Gradient Direction
    // ---------------------------------------------------------
    uint8_t* direction_data = (uint8_t*)aligned_alloc(32, size);
    if (!direction_data) {
        std::cerr << "Error: Memory allocation for Direction failed!" << std::endl;
        return 1;
    }
    gradient_direction_scalar(grad_x, grad_y, direction_data, width, height);
    std::cout << "Gradient Direction calculated successfully!" << std::endl; 

#ifndef __riscv
    std::cout << "--- Running on HOST: Testing File I/O ---" << std::endl;
    
    write_raw_image(filename, dummy_data, width, height);
    std::cout << "Successfully saved dummy image." << std::endl;

    uint8_t* read_data = read_raw_image(filename, width, height);
    if (read_data) {
        std::cout << "Successfully read image back." << std::endl;
        free(read_data);
    } else {
        std::cerr << "Failed to read image!" << std::endl;
    }
#else
    std::cout << "--- Running on RISC-V (QEMU): Skipping File I/O ---" << std::endl;
    std::cout << "Memory allocation and setup successful!" << std::endl;
#endif


    free(dummy_data);
    free(blurred_data);
    free(grad_x);
    free(grad_y);

    std::cout << "Phase completed successfully!" << std::endl;

    free(magnitude_data);
    free(direction_data);

    return 0;
}
