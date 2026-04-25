#include <iostream>
#include <cstdlib>
#include "../include/image_io.h"

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
    std::cout << "Phase completed successfully!" << std::endl;
    
    return 0;
}