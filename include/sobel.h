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

                    // Clamp out-of-bounds coordinates
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

#ifdef __riscv
#include <riscv_vector.h>

void sobel_gradients_rvv(const uint8_t* input,
                         int16_t* grad_x,
                         int16_t* grad_y,
                         int width, int height) {
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x += 8) {
            size_t vl = __riscv_vsetvl_e8m1(width - x);

            // Load 3 rows around (y,x)
            const uint8_t* row_above = &input[(y-1)*width + (x-1)];
            const uint8_t* row_mid   = &input[y*width + (x-1)];
            const uint8_t* row_below = &input[(y+1)*width + (x-1)];

            vuint8m1_t top = __riscv_vle8_v_u8m1(row_above, vl+2);
            vuint8m1_t mid = __riscv_vle8_v_u8m1(row_mid, vl+2);
            vuint8m1_t bot = __riscv_vle8_v_u8m1(row_below, vl+2);

            // Widen to 16-bit
            vuint16m2_t top_u16 = __riscv_vzext_vf2_u16m2(top, vl);
            vuint16m2_t mid_u16 = __riscv_vzext_vf2_u16m2(mid, vl);
            vuint16m2_t bot_u16 = __riscv_vzext_vf2_u16m2(bot, vl);

            vint16m2_t top16 = __riscv_vreinterpret_v_u16m2_i16m2(top_u16);
            vint16m2_t mid16 = __riscv_vreinterpret_v_u16m2_i16m2(mid_u16);
            vint16m2_t bot16 = __riscv_vreinterpret_v_u16m2_i16m2(bot_u16);

            // Shift left/right to get neighbors
            vint16m2_t top_left   = __riscv_vslide1up_vx_i16m2(top16, 0, vl);
            vint16m2_t top_right  = __riscv_vslide1down_vx_i16m2(top16, 0, vl);
            vint16m2_t mid_left   = __riscv_vslide1up_vx_i16m2(mid16, 0, vl);
            vint16m2_t mid_right  = __riscv_vslide1down_vx_i16m2(mid16, 0, vl);
            vint16m2_t bot_left   = __riscv_vslide1up_vx_i16m2(bot16, 0, vl);
            vint16m2_t bot_right  = __riscv_vslide1down_vx_i16m2(bot16, 0, vl);

            // Sobel Gx
            vint16m2_t gx = __riscv_vmul_vx_i16m2(top_right, 1, vl);
            gx = __riscv_vmacc_vx_i16m2(gx, 2, mid_right, vl);
            gx = __riscv_vmacc_vx_i16m2(gx, 1, bot_right, vl);
            gx = __riscv_vmacc_vx_i16m2(gx, -1, top_left, vl);
            gx = __riscv_vmacc_vx_i16m2(gx, -2, mid_left, vl);
            gx = __riscv_vmacc_vx_i16m2(gx, -1, bot_left, vl);

            // Sobel Gy
            vint16m2_t gy = __riscv_vmul_vx_i16m2(bot_left, 1, vl);
            gy = __riscv_vmacc_vx_i16m2(gy, 2, bot16, vl);
            gy = __riscv_vmacc_vx_i16m2(gy, 1, bot_right, vl);
            gy = __riscv_vmacc_vx_i16m2(gy, -1, top_left, vl);
            gy = __riscv_vmacc_vx_i16m2(gy, -2, top16, vl);
            gy = __riscv_vmacc_vx_i16m2(gy, -1, top_right, vl);

            // Store results
            __riscv_vse16_v_i16m2(&grad_x[y*width + x], gx, vl);
            __riscv_vse16_v_i16m2(&grad_y[y*width + x], gy, vl);
        }
    }
}
#endif


#endif

