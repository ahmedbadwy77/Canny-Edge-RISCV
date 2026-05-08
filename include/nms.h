#ifndef NMS_H
#define NMS_H

#include <cstdint>

void non_max_suppression(const uint8_t* magnitude, const uint8_t* direction,
                          uint8_t* output, int width, int height) {
    // Zero out the border
    for (int x = 0; x < width; x++) {
        output[x] = 0;
        output[(height-1)*width+x] = 0;
    }
    for (int y = 0; y < height; y++) {
        output[y*width] = 0;
        output[y*width + width-1] = 0;
    }

    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            int idx = y * width + x;
            uint8_t mag = magnitude[idx];
            uint8_t dir = direction[idx];
            uint8_t n1, n2;  // the two neighbours to compare against

            if (dir == 0) {
                // Horizontal edge — compare left and right
                n1 = magnitude[idx - 1];
                n2 = magnitude[idx + 1];
            } else if (dir == 90) {
                // Vertical edge — compare top and bottom
                n1 = magnitude[idx - width];
                n2 = magnitude[idx + width];
            } else if (dir == 45) {
                // Positive diagonal — compare top-right and bottom-left
                n1 = magnitude[idx - width + 1];
                n2 = magnitude[idx + width - 1];
            } else {
                // Negative diagonal (135) — compare top-left and bottom-right
                n1 = magnitude[idx - width - 1];
                n2 = magnitude[idx + width + 1];
            }

            // Keep only if this pixel is the local maximum
            output[idx] = (mag >= n1 && mag >= n2) ? mag : 0;
        }
    }
}

#endif