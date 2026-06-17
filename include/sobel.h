#ifndef SOBEL_H
#define SOBEL_H

#include <cstdint>
#include <cmath>
#include <cstddef>

// Sobel-X kernel: detects vertical edges (horizontal intensity gradient)
const int8_t SOBEL_X[3][3] = {
    {-1, 0, 1},
    {-2, 0, 2},
    {-1, 0, 1}
};

// Sobel-Y kernel: detects horizontal edges (vertical intensity gradient)
const int8_t SOBEL_Y[3][3] = {
    { 1,  2,  1},
    { 0,  0,  0},
    {-1, -2, -1}
};

// Scalar baseline: templated to allow easy comparison with RVV version.
// Outputs SoA (Structure of Arrays) layout: separate grad_x and grad_y buffers.
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

#if defined(__riscv_vector)
#include <riscv_vector.h>

static inline vint16m2_t load_u8_as_i16m2(const uint8_t* ptr, size_t vl) {
    // vle8 loads vl adjacent 8-bit pixels. LMUL=m1 is enough for the byte
    // source vector; changing VLEN only changes how many pixels vl contains.
    vuint8m1_t pixels = __riscv_vle8_v_u8m1(ptr, vl);

    // vzext widens u8m1 to u16m2. LMUL becomes m2 because widening doubles
    // the register group; at larger VLEN the same code processes more pixels.
    vuint16m2_t pixels_u16 = __riscv_vzext_vf2_u16m2(pixels, vl);

    // Reinterpret keeps the same bits and LMUL=m2, but treats values as signed
    // int16 for Sobel multiply-accumulate. VLEN does not change the semantics.
    return __riscv_vreinterpret_v_u16m2_i16m2(pixels_u16);
}

void sobel_gradients_rvv(const uint8_t* input,
                         int16_t* grad_x,
                         int16_t* grad_y,
                         int width, int height) {
    if (width <= 2 || height <= 2) {
        sobel_gradients_scalar(input, grad_x, grad_y, width, height);
        return;
    }

    // Match the scalar clamped-border behavior. The interior is overwritten
    // by the vector loop below.
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            if (y > 0 && y < height - 1 && x > 0 && x < width - 1) {
                continue;
            }

            int32_t sum_x = 0;
            int32_t sum_y = 0;
            for (int ky = -1; ky <= 1; ++ky) {
                for (int kx = -1; kx <= 1; ++kx) {
                    int nx = x + kx;
                    int ny = y + ky;
                    if (nx < 0) nx = 0;
                    else if (nx >= width) nx = width - 1;
                    if (ny < 0) ny = 0;
                    else if (ny >= height) ny = height - 1;

                    uint8_t pixel = input[ny * width + nx];
                    sum_x += pixel * SOBEL_X[ky + 1][kx + 1];
                    sum_y += pixel * SOBEL_Y[ky + 1][kx + 1];
                }
            }
            int idx = y * width + x;
            grad_x[idx] = static_cast<int16_t>(sum_x);
            grad_y[idx] = static_cast<int16_t>(sum_y);
        }
    }

    for (int y = 1; y < height - 1; ++y) {
        for (int x = 1; x < width - 1; ) {
            // vsetvl chooses the largest legal vl for the remaining row tail.
            // e8m1 is used because source pixels are bytes; wider VLEN values
            // simply increase vl and reduce the number of loop iterations.
            size_t vl = __riscv_vsetvl_e8m1((width - 1) - x);
            int idx = y * width + x;

            const uint8_t* top = &input[(y - 1) * width + x];
            const uint8_t* mid = &input[y * width + x];
            const uint8_t* bot = &input[(y + 1) * width + x];

            // Each load below fetches one aligned Sobel neighborhood column
            // for vl adjacent output pixels. The helper widens to i16m2 so the
            // following signed coefficients fit without overflow.
            vint16m2_t tl = load_u8_as_i16m2(top - 1, vl);
            vint16m2_t tc = load_u8_as_i16m2(top, vl);
            vint16m2_t tr = load_u8_as_i16m2(top + 1, vl);
            vint16m2_t ml = load_u8_as_i16m2(mid - 1, vl);
            vint16m2_t mr = load_u8_as_i16m2(mid + 1, vl);
            vint16m2_t bl = load_u8_as_i16m2(bot - 1, vl);
            vint16m2_t bc = load_u8_as_i16m2(bot, vl);
            vint16m2_t br = load_u8_as_i16m2(bot + 1, vl);

            // vmul starts Gx with the +1 right-top term. i16m2 matches the
            // widened pixel type and gives enough range for Sobel output;
            // changing VLEN changes only vl, not the kernel math.
            vint16m2_t gx = __riscv_vmul_vx_i16m2(tr, 1, vl);
            // vmacc accumulates scalar Sobel-X coefficients into gx:
            // +2*mr, +1*br, -1*tl, -2*ml, -1*bl. LMUL stays m2 throughout.
            gx = __riscv_vmacc_vx_i16m2(gx, 2, mr, vl);
            gx = __riscv_vmacc_vx_i16m2(gx, 1, br, vl);
            gx = __riscv_vmacc_vx_i16m2(gx, -1, tl, vl);
            gx = __riscv_vmacc_vx_i16m2(gx, -2, ml, vl);
            gx = __riscv_vmacc_vx_i16m2(gx, -1, bl, vl);

            // vmul starts Gy with the +1 left-top term. The same i16m2 LMUL
            // choice is used because Gy has the same output range as Gx.
            vint16m2_t gy = __riscv_vmul_vx_i16m2(tl, 1, vl);
            // vmacc accumulates scalar Sobel-Y coefficients:
            // +2*tc, +1*tr, -1*bl, -2*bc, -1*br. Larger VLEN only means
            // more pixels are accumulated in this same strip-mined iteration.
            gy = __riscv_vmacc_vx_i16m2(gy, 2, tc, vl);
            gy = __riscv_vmacc_vx_i16m2(gy, 1, tr, vl);
            gy = __riscv_vmacc_vx_i16m2(gy, -1, bl, vl);
            gy = __riscv_vmacc_vx_i16m2(gy, -2, bc, vl);
            gy = __riscv_vmacc_vx_i16m2(gy, -1, br, vl);

            // Store vl int16 Sobel outputs. m2 matches gx/gy; different VLEN
            // values only affect how many elements are written per iteration.
            __riscv_vse16_v_i16m2(&grad_x[idx], gx, vl);
            __riscv_vse16_v_i16m2(&grad_y[idx], gy, vl);

            x += vl;
        }
    }
}
#endif

#endif

