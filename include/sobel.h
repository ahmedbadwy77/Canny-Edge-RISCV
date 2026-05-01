#ifndef SOBEL_H
#define SOBEL_H

#include <cstdint>
#include <cmath>

const int8_t SOBEL_X[3][3] = {
    {-1, 0, 1},
    {-2, 0, 2},
    {-1, 0, 1}
};

const int8_t SOBEL_Y[3][3] = {
    { 1,  2,  1},
    { 0,  0,  0},
    {-1, -2, -1}
};

template<typename PixelT = uint8_t, typename GradT = int16_t>
void sobel_gradients_scalar(const PixelT* input, GradT* grad_x, GradT* grad_y, int width, int height) {
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int32_t sum_x = 0;
            int32_t sum_y = 0;
            for (int ky = -1; ky <= 1; ++ky) {
                for (int kx = -1; kx <= 1; ++kx) {
                    int nx = x + kx;
                    int ny = y + ky;
                    
                    // Clamp out-of-bounds coordinates to the image edge
                    if (nx < 0) nx = 0;
                    else if (nx >= width) nx = width - 1;
                    
                    if (ny < 0) ny = 0;
                    else if (ny >= height) ny = height - 1;

                    PixelT pixel_val = input[ny * width + nx];
                    sum_x += pixel_val * SOBEL_X[ky + 1][kx + 1];
                    sum_y += pixel_val * SOBEL_Y[ky + 1][kx + 1];
                }
            }
            grad_x[y * width + x] = (GradT)sum_x;
            grad_y[y * width + x] = (GradT)sum_y;
        }
    }
}

#endif