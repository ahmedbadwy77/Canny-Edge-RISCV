#ifndef GAUSSIAN_H
#define GAUSSIAN_H

#include <cstdint>

// 5x5 Gaussian kernel with integer coefficients (sum = 273, sigma ≈ 1.0)
// Using integer arithmetic throughout avoids floating-point on embedded targets.
const int16_t GAUSSIAN_KERNEL[5][5] = {
    { 1,  4,  7,  4,  1 },
    { 4, 16, 26, 16,  4 },
    { 7, 26, 41, 26,  7 },
    { 4, 16, 26, 16,  4 },
    { 1,  4,  7,  4,  1 }
};

// Scalar baseline: generic over pixel type, accumulator type, and kernel type.
// Template specialization allows the RVV version below to share the same interface.
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

// ============================================================
// RVV Optimized Gaussian Blur (Phase 6)
// Key ideas: strip-mining, data widening chain u8→u16→i32,
// LMUL=4 accumulator, fixed-point division instead of dividing by 273.
// ============================================================
#ifdef __riscv
#include <riscv_vector.h>

void gaussian_blur_rvv(const uint8_t* input, uint8_t* output, int width, int height) {
    // Zero-initialize output so border pixels (within 2px of all edges) remain black.
    // The RVV loop only processes interior pixels to avoid boundary checks in the
    // hot path — a deliberate performance tradeoff documented in the hints guide.
    for(int i = 0; i < width * height; i++) output[i] = 0;

    for (int y = 2; y < height - 2; ++y) {
        int vl;

        // ---------------------------------------------------------------
        // STRIP-MINING LOOP
        // RVV is vector-length-agnostic (VLA): we never hardcode how many
        // elements fit in a register. vsetvl returns vl at runtime based on
        // VLEN and LMUL. We advance x by vl each iteration until row is done.
        // The same binary runs correctly at VLEN=128, 256, and 512.
        // ---------------------------------------------------------------
        for (int x = 2; x < width - 2; x += vl) {

            // __riscv_vsetvl_e8m1(n)
            // WHAT: Asks hardware "how many uint8 (e8) elements fit in one
            //       vector register group with LMUL=1 (m1)?" Returns that count as vl.
            // WHY e8m1: Input pixels are uint8. LMUL=1 gives us 32 logical registers,
            //           leaving room for the many temporaries in this 5x5 kernel loop.
            // VLEN EFFECT: VLEN=128 → vl=16. VLEN=256 → vl=32. VLEN=512 → vl=64.
            //              Code is IDENTICAL at all sizes — this is RVV's key advantage.
            vl = __riscv_vsetvl_e8m1(width - 2 - x);

            // __riscv_vmv_v_x_i32m4(0, vl)
            // WHAT: Broadcasts scalar 0 into every lane of a 32-bit signed vector
            //       register group (LMUL=4), initializing the running accumulator.
            // WHY i32m4: Worst-case accumulation across 25 kernel positions:
            //            255 * 41 * 25 = 261,375 — requires 32 bits (>16-bit max=65535).
            //            LMUL=4 is mandatory because the widening multiply chain below
            //            (u8m1→u16m2→i32m4) always doubles LMUL at each stage.
            // VLEN EFFECT: More elements per group at wider VLEN; logic unchanged.
            vint32m4_t vsum = __riscv_vmv_v_x_i32m4(0, vl);

            // Scalar loops over 5x5 kernel positions; inner vector ops process vl pixels at once
            for (int ky = -2; ky <= 2; ++ky) {
                for (int kx = -2; kx <= 2; ++kx) {

                    // __riscv_vle8_v_u8m1(ptr, vl)
                    // WHAT: Vector Load Elements (8-bit) — loads vl consecutive uint8
                    //       pixels from memory into a LMUL=1 vector register group.
                    // WHY u8m1: Source pixels are 1 byte each; LMUL=1 matches vsetvl above.
                    // VLEN EFFECT: More pixels loaded per call at wider VLEN.
                    vuint8m1_t vpix = __riscv_vle8_v_u8m1(&input[(y + ky) * width + (x + kx)], vl);

                    // __riscv_vzext_vf2_u16m2(vpix, vl)
                    // WHAT: Zero-EXTend by Factor 2 — widens each uint8 element to uint16.
                    //       Output register group is LMUL=2 (m2) because element size doubled.
                    // WHY: We need 16-bit values to feed the widening multiply below without
                    //      intermediate overflow. Zero-extension is correct for unsigned pixels.
                    // CRITICAL LMUL CHAIN: u8m1 → vzext_vf2 → u16m2 (LMUL doubled).
                    //      This chain is mandatory and must be tracked carefully.
                    vuint16m2_t vpix_u16 = __riscv_vzext_vf2_u16m2(vpix, vl);

                    // __riscv_vreinterpret_v_u16m2_i16m2(vpix_u16)
                    // WHAT: Reinterprets the raw bits of a uint16 vector as int16.
                    //       Zero data movement or conversion — purely a type-system cast.
                    // WHY: vwmul (below) requires signed int16 inputs. Pixel values 0-255
                    //      all fit in int16 (max=32767), so reinterpretation is safe.
                    vint16m2_t vpix_i16 = __riscv_vreinterpret_v_u16m2_i16m2(vpix_u16);

                    // __riscv_vwmul_vx_i32m4(vpix_i16, coeff, vl)
                    // WHAT: Widening Multiply (vector × scalar) — multiplies each int16
                    //       element by scalar coeff, producing int32 results (LMUL=4).
                    // WHY vwmul: A single instruction replaces explicit widen+multiply.
                    //            Kernel coefficients (1–41) are all positive; signed multiply
                    //            is correct and produces no unexpected sign extension.
                    // CRITICAL LMUL CHAIN: i16m2 → vwmul_vx → i32m4 (LMUL doubled again).
                    //      This is why vsum MUST be declared as i32m4, not i32m2.
                    int16_t coeff = GAUSSIAN_KERNEL[ky + 2][kx + 2];
                    vint32m4_t vprod = __riscv_vwmul_vx_i32m4(vpix_i16, coeff, vl);

                    // __riscv_vadd_vv_i32m4(vsum, vprod, vl)
                    // WHAT: Vector Add (element-wise) — adds vprod into vsum lane by lane.
                    // WHY: Accumulates all 25 kernel multiply-results in 32-bit precision
                    //      before the final normalization step, preventing overflow.
                    vsum = __riscv_vadd_vv_i32m4(vsum, vprod, vl);
                }
            }

            // --- Fast Fixed-Point Division replacing (sum / 273) ---

            // __riscv_vmul_vx_i32m4(vsum, 240, vl)
            // WHAT: Multiplies every 32-bit accumulator element by scalar 240.
            // WHY: This is Step 1 of the fixed-point trick: (sum * 240) >> 16 ≈ sum / 273.
            //      Derivation: 65536 / 273 ≈ 240.06, so multiply by 240 then shift 16.
            //      Integer division is expensive on embedded CPUs; this eliminates it.
            //      Maximum precision error: < 0.1%, negligible for 8-bit pixel output.
            vsum = __riscv_vmul_vx_i32m4(vsum, 240, vl);

            // __riscv_vsra_vx_i32m4(vsum, 16, vl)
            // WHAT: Vector Shift Right Arithmetic by 16 bits — completes the fixed-point
            //       division. Arithmetic (not logical) shift preserves sign for negatives.
            // WHY: Divides by 2^16=65536, completing the approximation of ÷273.
            vsum = __riscv_vsra_vx_i32m4(vsum, 16, vl);

            // __riscv_vmax_vx_i32m4(vsum, 0, vl)
            // WHAT: Element-wise maximum with scalar 0 — clamps negatives to zero.
            // WHY: Fixed-point rounding can produce small negatives. This replaces
            //      a conditional branch (if sum < 0) with a branchless vector op.
            vsum = __riscv_vmax_vx_i32m4(vsum, 0, vl);

            // __riscv_vmin_vx_i32m4(vsum, 255, vl)
            // WHAT: Element-wise minimum with scalar 255 — clamps values above 255.
            // WHY: Saturation to valid 8-bit pixel range. Together with vmax, these
            //      two instructions replace two conditional branches with no jumps.
            vsum = __riscv_vmin_vx_i32m4(vsum, 255, vl);

            // --- Narrowing chain: i32m4 → u32m4 → u16m2 → u8m1 ---

            // __riscv_vreinterpret_v_i32m4_u32m4(vsum)
            // WHAT: Reinterprets int32 vector bits as uint32. No data conversion.
            // WHY: vncvt (narrowing convert) requires unsigned input to produce unsigned
            //      output. Values are already in [0,255] so reinterpret is safe and correct.
            vuint32m4_t vsum_u32 = __riscv_vreinterpret_v_i32m4_u32m4(vsum);

            // __riscv_vncvt_x_x_w_u16m2(vsum_u32, vl)
            // WHAT: Narrowing Convert — halves element width from uint32 to uint16.
            //       Output LMUL is also halved: m4 → m2.
            // WHY: First step of 3-stage narrowing chain to reach uint8 output.
            //      No data loss: values are clamped to [0,255], well within uint16 range.
            // CRITICAL LMUL CHAIN: u32m4 → vncvt → u16m2 (LMUL halved).
            vuint16m2_t vsum_u16 = __riscv_vncvt_x_x_w_u16m2(vsum_u32, vl);

            // __riscv_vncvt_x_x_w_u8m1(vsum_u16, vl)
            // WHAT: Second narrowing — halves uint16 to uint8. LMUL: m2 → m1.
            // WHY: Produces the final 8-bit pixel values for the output image.
            // CRITICAL LMUL CHAIN: u16m2 → vncvt → u8m1 (LMUL halved again).
            //      Full LMUL journey: u8m1→u16m2→i16m2→i32m4→u32m4→u16m2→u8m1
            vuint8m1_t vsum_u8 = __riscv_vncvt_x_x_w_u8m1(vsum_u16, vl);

            // __riscv_vse8_v_u8m1(ptr, vsum_u8, vl)
            // WHAT: Vector Store Elements (8-bit) — writes vl uint8 pixels to consecutive
            //       memory locations starting at ptr.
            // WHY u8m1: Matches the final narrowed output type from the chain above.
            // VLEN EFFECT: More pixels stored per call at wider VLEN. The pointer
            //              advances by vl each strip-mining iteration automatically.
            __riscv_vse8_v_u8m1(&output[y * width + x], vsum_u8, vl);
        }
    }
}
#endif

#endif
