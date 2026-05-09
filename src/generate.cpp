#include <iostream>
#include <cstdlib>
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

    // Read the image
    uint8_t* input_buf = read_raw_image("input.raw", width, height);
    if (!input_buf) {
        std::cout << "Error: Could not read input.raw\n";
        return 1;
    }

    // Allocate memory
    uint8_t* blurred_data = (uint8_t*)malloc(size);
    int16_t* grad_x = (int16_t*)malloc(size * sizeof(int16_t));
    int16_t* grad_y = (int16_t*)malloc(size * sizeof(int16_t));
    uint8_t* magnitude_data = (uint8_t*)malloc(size);
    uint8_t* direction_data = (uint8_t*)malloc(size);
    uint8_t* nms_data = (uint8_t*)malloc(size);
    uint8_t* threshold_data = (uint8_t*)malloc(size);

    // Run the pipeline ONCE
    gaussian_blur_scalar(input_buf, blurred_data, width, height);
    sobel_gradients_scalar(blurred_data, grad_x, grad_y, width, height);
    gradient_magnitude_scalar(grad_x, grad_y, magnitude_data, width, height, MagMethod::L1);
    gradient_direction_scalar(grad_x, grad_y, direction_data, width, height);
    non_max_suppression(magnitude_data, direction_data, nms_data, width, height);
    
    uint8_t low, high;
    auto_threshold(nms_data, size, low, high);
    double_threshold(nms_data, threshold_data, width, height, low, high);
    hysteresis_tracing(threshold_data, width, height);

    // SAVE THE OUTPUT IMAGE!
    FILE* out_file = fopen("output.raw", "wb");
    if(out_file) {
        fwrite(threshold_data, 1, size, out_file);
        fclose(out_file);
        std::cout << "Success! Saved edge detection to output.raw\n";
    }

    return 0;
}
