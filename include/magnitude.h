#ifndef MAGNITUDE_H
#define MAGNITUDE_H

#include <cstdint>
#include <cmath>
#include <algorithm>
#include <cstdlib>

// Enum for magnitude method
enum class MagMethod {
    L1,
    L2
};

template<typename GradT = int16_t, typename PixelT = uint8_t>
void gradient_magnitude_scalar(const GradT* grad_x, const GradT* grad_y, PixelT* output, int width, int height, MagMethod method) {
    int size = width * height;
    
    // Allocate temporary buffer for raw magnitude (uint32_t to avoid overflow)
    uint32_t* raw_mag = (uint32_t*)aligned_alloc(32, size * sizeof(uint32_t));
    uint32_t max_mag = 0;

    // Pass 1: Calculate magnitude and find max value
    for (int i = 0; i < size; ++i) {
        uint32_t mag = 0;
        if (method == MagMethod::L1) {
            mag = std::abs(grad_x[i]) + std::abs(grad_y[i]);
        } else {
            // L2: sqrt(Gx^2 + Gy^2)
            mag = (uint32_t)std::round(std::sqrt(grad_x[i] * grad_x[i] + grad_y[i] * grad_y[i]));
        }
        
        raw_mag[i] = mag;
        
        if (mag > max_mag) {
            max_mag = mag;
        }
    }

    // Pass 2: Normalize to [0, 255]
    if (max_mag == 0) {
        for (int i = 0; i < size; ++i) {
            output[i] = 0;
        }
    } else {
        for (int i = 0; i < size; ++i) {
            output[i] = (PixelT)((raw_mag[i] * 255) / max_mag);
        }
    }

    free(raw_mag);
}

#endif
