#ifndef GAUSSIAN_H
#define GAUSSIAN_H

#include <cstdint>

// 5x5 Gaussian kernel
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

#if defined(__riscv_vector)
#include <riscv_vector.h>

void gaussian_blur_rvv(const uint8_t* input, uint8_t* output, int width, int height) {
    if (width <= 4 || height <= 4) {
        gaussian_blur_scalar(input, output, width, height);
        return;
    }

    // The vector loop only handles pixels with a complete 5x5 neighborhood.
    // Compute the two-pixel border with the scalar clamping rules so both
    // implementations produce identical output.
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            if (y >= 2 && y < height - 2 && x >= 2 && x < width - 2) {
                continue;
            }

            uint32_t sum = 0;
            for (int ky = -2; ky <= 2; ++ky) {
                for (int kx = -2; kx <= 2; ++kx) {
                    int nx = x + kx;
                    int ny = y + ky;
                    if (nx < 0) nx = 0;
                    else if (nx >= width) nx = width - 1;
                    if (ny < 0) ny = 0;
                    else if (ny >= height) ny = height - 1;
                    sum += input[ny * width + nx] * GAUSSIAN_KERNEL[ky + 2][kx + 2];
                }
            }
            output[y * width + x] = static_cast<uint8_t>(sum / 273);
        }
    }

    for (int y = 2; y < height - 2; ++y) {
        int vl;
        for (int x = 2; x < width - 2; x += vl) {
            vl = __riscv_vsetvl_e8m1(width - 2 - x);

            // Use UNSIGNED 32-bit accumulator (LMUL=4)
            vuint32m4_t vsum = __riscv_vmv_v_x_u32m4(0, vl);

            for (int ky = -2; ky <= 2; ++ky) {
                for (int kx = -2; kx <= 2; ++kx) {
                    uint16_t coeff = GAUSSIAN_KERNEL[ky + 2][kx + 2];
                    vuint8m1_t vpix = __riscv_vle8_v_u8m1(&input[(y + ky) * width + (x + kx)], vl);
                    
                    // Zero-extend to 16-bit
                    vuint16m2_t vpix_u16 = __riscv_vzext_vf2_u16m2(vpix, vl);
                    
                    // Unsigned Widening Multiply-Accumulate (3 instructions reduced to 1!)
                    vsum = __riscv_vwmaccu_vx_u32m4(vsum, coeff, vpix_u16, vl);
                }
            }

            // Match the scalar integer division exactly.
            vsum = __riscv_vdivu_vx_u32m4(vsum, 273, vl);
            vsum = __riscv_vminu_vx_u32m4(vsum, 255, vl); // Clamp max to 255

            // Narrowing chain: u32m4 -> u16m2 -> u8m1
            vuint16m2_t vsum_u16 = __riscv_vncvt_x_x_w_u16m2(vsum, vl);
            vuint8m1_t vsum_u8 = __riscv_vncvt_x_x_w_u8m1(vsum_u16, vl);

            __riscv_vse8_v_u8m1(&output[y * width + x], vsum_u8, vl);
        }
    }
}
#endif

#endif
