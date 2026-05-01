#ifndef GAUSSIAN_H
#define GAUSSIAN_H

#include <cstdint>

// مصفوفة الفلتر 5x5 (مجموعها 273)
const int16_t GAUSSIAN_KERNEL[5][5] = {
    { 1,  4,  7,  4,  1 },
    { 4, 16, 26, 16,  4 },
    { 7, 26, 41, 26,  7 },
    { 4, 16, 26, 16,  4 },
    { 1,  4,  7,  4,  1 }
};

template<typename PixelT = uint8_t, typename AccT = int32_t, typename KernelT = int16_t>
void gaussian_blur_scalar(const PixelT* input, PixelT* output, int width, int height) {
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            AccT sum = 0;
            for (int ky = -2; ky <= 2; ++ky) {
                for (int kx = -2; kx <= 2; ++kx) {
                    int nx = x + kx;
                    int ny = y + ky;
                    
                    // Clamp out-of-bounds coordinates to the image edge
                    if (nx < 0) nx = 0;
                    else if (nx >= width) nx = width - 1;
                    
                    if (ny < 0) ny = 0;
                    else if (ny >= height) ny = height - 1;

                    sum += input[ny * width + nx] * GAUSSIAN_KERNEL[ky + 2][kx + 2];
                }
            }
            sum /= 273;
            if (sum < 0) sum = 0;
            if (sum > 255) sum = 255;
            output[y * width + x] = (PixelT)sum;
        }
    }
}

#endif