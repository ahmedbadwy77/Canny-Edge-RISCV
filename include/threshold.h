#ifndef THRESHOLD_H
#define THRESHOLD_H

#include <cstdint>

void double_threshold(const uint8_t* nms_output, uint8_t* output,
                       int width, int height,
                       uint8_t low_thresh, uint8_t high_thresh) {

    const uint8_t STRONG = 255;
    const uint8_t WEAK   = 128;

    int size = width * height;

    for (int i = 0; i < size; i++) {
        uint8_t val = nms_output[i];
        if (val >= high_thresh) {
            output[i] = STRONG;
        } else if (val >= low_thresh) {
            output[i] = WEAK;
        } else {
            output[i] = 0;
        }
    }
}

// Automatically pick low and high thresholds based on the max magnitude value
void auto_threshold(const uint8_t* nms_output, size_t size,
                    uint8_t& low_out, uint8_t& high_out,
                    float low_ratio = 0.10f, float high_ratio = 0.20f) {
    uint8_t max_val = 0;
    for (size_t i = 0; i < size; i++)
        if (nms_output[i] > max_val) max_val = nms_output[i];

    high_out = (uint8_t)(max_val * high_ratio);
    low_out  = (uint8_t)(max_val * low_ratio);
}

#endif