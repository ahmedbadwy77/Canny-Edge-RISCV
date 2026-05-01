#ifndef DIRECTION_H
#define DIRECTION_H

#include <cstdint>
#include <cstdlib>

template<typename GradT = int16_t>
void gradient_direction_scalar(const GradT* grad_x, const GradT* grad_y, uint8_t* dir_output, int width, int height) {
    int size = width * height;
    
    for (int i = 0; i < size; ++i) {
        GradT dx = grad_x[i];
        GradT dy = grad_y[i];

        if (dx == 0 && dy == 0) {
            dir_output[i] = 0; // Default to 0 degrees
            continue;
        }

        int32_t abs_dx = std::abs(dx);
        int32_t abs_dy = std::abs(dy);

        // Integer approximation for angle bounds to avoid atan2()
        // tan(22.5) ~= 0.414 -> abs_dy * 1000 <= abs_dx * 414
        // tan(67.5) ~= 2.414 -> abs_dy * 1000 >  abs_dx * 2414

        if (abs_dy * 1000 <= abs_dx * 414) {
            dir_output[i] = 0;   // Horizontal
        } else if (abs_dy * 1000 > abs_dx * 2414) {
            dir_output[i] = 90;  // Vertical
        } else {
            // Diagonal
            if ((dx > 0 && dy > 0) || (dx < 0 && dy < 0)) {
                dir_output[i] = 45;  // Positive diagonal
            } else {
                dir_output[i] = 135; // Negative diagonal
            }
        }
    }
}

#endif
