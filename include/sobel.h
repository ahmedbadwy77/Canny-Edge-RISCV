#ifndef SOBEL_H
#define SOBEL_H

#include <cstdint>
#include <cmath>

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
// SoA chosen over AoS because vector loads of consecutive Gx values need no gather.
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

// ============================================================
// RVV Optimized Sobel Gradients (Phase 6)
// Key ideas: load 3 rows simultaneously, use vslide1up/down to
// get left/right neighbors, apply Sobel coefficients with vmacc.
// ============================================================
#ifdef __riscv
#include <riscv_vector.h>

void sobel_gradients_rvv(const uint8_t* input,
                         int16_t* grad_x,
                         int16_t* grad_y,
                         int width, int height) {
    // Process interior pixels only (1-pixel border excluded) to avoid
    // boundary checks inside the vectorized inner loop.
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x += 8) {

            // __riscv_vsetvl_e8m1(width - x)
            // WHAT: Sets vector length for uint8 (e8) elements with LMUL=1 (m1).
            //       Returns vl = min(width - x, VLEN/8) elements to process.
            // WHY e8m1: Input pixels are uint8. LMUL=1 chosen because after widening
            //           to u16m2 we still have enough registers for 6 row vectors
            //           (top/mid/bot each needing left/center/right variants).
            // VLEN EFFECT: VLEN=128 → vl≤16. VLEN=256 → vl≤32. VLEN=512 → vl≤64.
            size_t vl = __riscv_vsetvl_e8m1(width - x);

            // Pointers to the three rows needed for the 3x3 Sobel kernel.
            // Starting at (x-1) to load left-neighbor pixels needed by vslide below.
            const uint8_t* row_above = &input[(y-1)*width + (x-1)];
            const uint8_t* row_mid   = &input[y*width + (x-1)];
            const uint8_t* row_below = &input[(y+1)*width + (x-1)];

            // __riscv_vle8_v_u8m1(ptr, vl+2)
            // WHAT: Loads vl+2 consecutive uint8 pixels from each row.
            // WHY vl+2: We load one extra pixel on each side (left and right neighbors)
            //           so that vslide1up/down can access all required neighbors
            //           without a separate boundary-handling load.
            // VLEN EFFECT: More pixels loaded per call at wider VLEN.
            vuint8m1_t top = __riscv_vle8_v_u8m1(row_above, vl+2);
            vuint8m1_t mid = __riscv_vle8_v_u8m1(row_mid, vl+2);
            vuint8m1_t bot = __riscv_vle8_v_u8m1(row_below, vl+2);

            // __riscv_vzext_vf2_u16m2(top, vl)
            // WHAT: Zero-extends each uint8 element to uint16, outputting LMUL=2 (m2).
            // WHY: Sobel multiply-accumulate operates on 16-bit values to prevent
            //      overflow. Max Sobel output for 8-bit input: 4*255=1020, fits in int16.
            // CRITICAL LMUL CHAIN: u8m1 → vzext_vf2 → u16m2 (LMUL doubled).
            vuint16m2_t top_u16 = __riscv_vzext_vf2_u16m2(top, vl);
            vuint16m2_t mid_u16 = __riscv_vzext_vf2_u16m2(mid, vl);
            vuint16m2_t bot_u16 = __riscv_vzext_vf2_u16m2(bot, vl);

            // __riscv_vreinterpret_v_u16m2_i16m2(top_u16)
            // WHAT: Reinterprets uint16 bits as int16. No data movement.
            // WHY: Sobel coefficients are signed (-1, -2). We need signed arithmetic
            //      for correct signed gradient computation. Pixel values 0-255 are
            //      all positive so reinterpretation introduces no sign error.
            vint16m2_t top16 = __riscv_vreinterpret_v_u16m2_i16m2(top_u16);
            vint16m2_t mid16 = __riscv_vreinterpret_v_u16m2_i16m2(mid_u16);
            vint16m2_t bot16 = __riscv_vreinterpret_v_u16m2_i16m2(bot_u16);

            // __riscv_vslide1up_vx_i16m2(top16, 0, vl)
            // WHAT: Slides vector elements UP by 1 position (toward index 0).
            //       Element[i] of output = Element[i+1] of input.
            //       The last element (vacated slot) is filled with scalar 0.
            // WHY: This gives us the "left neighbor" of each pixel position without
            //      a separate gather or scalar loop — a key vectorization technique.
            //      Sobel-X needs pixel[x-1] and pixel[x+1] for each output pixel.
            // VLEN EFFECT: The slide distance is always 1 element regardless of VLEN.
            vint16m2_t top_left   = __riscv_vslide1up_vx_i16m2(top16, 0, vl);
            vint16m2_t mid_left   = __riscv_vslide1up_vx_i16m2(mid16, 0, vl);
            vint16m2_t bot_left   = __riscv_vslide1up_vx_i16m2(bot16, 0, vl);

            // __riscv_vslide1down_vx_i16m2(top16, 0, vl)
            // WHAT: Slides vector elements DOWN by 1 (toward higher index).
            //       Element[i] of output = Element[i-1] of input.
            //       The first element (vacated slot) is filled with scalar 0.
            // WHY: Gives us the "right neighbor" of each pixel for Sobel-X computation.
            //      Together with vslide1up, we can compute all horizontal neighbors
            //      from a single loaded row — no scatter/gather needed.
            vint16m2_t top_right  = __riscv_vslide1down_vx_i16m2(top16, 0, vl);
            vint16m2_t mid_right  = __riscv_vslide1down_vx_i16m2(mid16, 0, vl);
            vint16m2_t bot_right  = __riscv_vslide1down_vx_i16m2(bot16, 0, vl);

            // --- Sobel Gx = -1*TL + 1*TR + -2*ML + 2*MR + -1*BL + 1*BR ---

            // __riscv_vmul_vx_i16m2(top_right, 1, vl)
            // WHAT: Multiplies each int16 element by scalar 1. Initializes gx accumulator
            //       with the top-right contribution (coefficient = +1 in Sobel-X).
            // WHY vmul to start: vmacc requires an existing accumulator. We initialize
            //     gx with the first non-zero term then accumulate the rest with vmacc.
            vint16m2_t gx = __riscv_vmul_vx_i16m2(top_right, 1, vl);

            // __riscv_vmacc_vx_i16m2(acc, scalar, vec, vl)
            // WHAT: Multiply-ACCumulate — computes acc = acc + (scalar * vec) element-wise.
            //       Single instruction replacing separate multiply and add.
            // WHY: Applies each Sobel-X coefficient to its corresponding neighbor row.
            //      Coefficients: mid_right=+2, bot_right=+1, top_left=-1, mid_left=-2, bot_left=-1.
            //      Negative coefficients are handled correctly by signed 16-bit arithmetic.
            // VLEN EFFECT: More pixels processed per call at wider VLEN; logic unchanged.
            gx = __riscv_vmacc_vx_i16m2(gx, 2, mid_right, vl);
            gx = __riscv_vmacc_vx_i16m2(gx, 1, bot_right, vl);
            gx = __riscv_vmacc_vx_i16m2(gx, -1, top_left, vl);
            gx = __riscv_vmacc_vx_i16m2(gx, -2, mid_left, vl);
            gx = __riscv_vmacc_vx_i16m2(gx, -1, bot_left, vl);

            // --- Sobel Gy = 1*BL + 2*B + 1*BR + -1*TL + -2*T + -1*TR ---
            // Same pattern as Gx but using top/bottom rows with different coefficients.
            vint16m2_t gy = __riscv_vmul_vx_i16m2(bot_left, 1, vl);
            gy = __riscv_vmacc_vx_i16m2(gy, 2, bot16, vl);
            gy = __riscv_vmacc_vx_i16m2(gy, 1, bot_right, vl);
            gy = __riscv_vmacc_vx_i16m2(gy, -1, top_left, vl);
            gy = __riscv_vmacc_vx_i16m2(gy, -2, top16, vl);
            gy = __riscv_vmacc_vx_i16m2(gy, -1, top_right, vl);

            // __riscv_vse16_v_i16m2(ptr, vec, vl)
            // WHAT: Vector Store Elements (16-bit) — writes vl int16 gradient values
            //       to consecutive memory locations starting at ptr.
            // WHY i16m2: Matches the computed gradient type. int16 is sufficient for
            //            Sobel: max value = 4*255 = 1020, well within int16 range (±32767).
            // SoA BENEFIT: Storing Gx and Gy in separate arrays (Structure of Arrays)
            //              means this store writes vl consecutive Gx values — a single
            //              sequential store with no striding or gather needed.
            __riscv_vse16_v_i16m2(&grad_x[y*width + x], gx, vl);
            __riscv_vse16_v_i16m2(&grad_y[y*width + x], gy, vl);
        }
    }
}
#endif

#endif
