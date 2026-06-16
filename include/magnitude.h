#ifndef MAGNITUDE_H
#define MAGNITUDE_H

#include <cstdint>
#include <cmath>
#include <algorithm>
#include <cstdlib>

// Enum for magnitude method selection
enum class MagMethod {
    L1,  // |Gx| + |Gy| — fast, integer-only, slight overestimate of diagonal edges
    L2   // sqrt(Gx^2 + Gy^2) — mathematically correct, requires floating point
};

// Scalar baseline: two-pass algorithm.
// Pass 1: compute raw magnitude for all pixels, track global max.
// Pass 2: normalize to [0,255] using the max value.
// Two passes are needed because normalization requires knowing the global max first.
template<typename GradT = int16_t, typename PixelT = uint8_t>
void gradient_magnitude_scalar(const GradT* grad_x, const GradT* grad_y, PixelT* output, int width, int height, MagMethod method) {
    int size = width * height;

    uint32_t* raw_mag = (uint32_t*)aligned_alloc(32, size * sizeof(uint32_t));
    uint32_t max_mag = 0;

    // Pass 1: Calculate magnitude and find max value
    for (int i = 0; i < size; ++i) {
        uint32_t mag = 0;
        if (method == MagMethod::L1) {
            mag = std::abs(grad_x[i]) + std::abs(grad_y[i]);
        } else {
            mag = (uint32_t)std::round(std::sqrt(grad_x[i] * grad_x[i] + grad_y[i] * grad_y[i]));
        }
        raw_mag[i] = mag;
        if (mag > max_mag) max_mag = mag;
    }

    // Pass 2: Normalize to [0, 255]
    if (max_mag == 0) {
        for (int i = 0; i < size; ++i) output[i] = 0;
    } else {
        for (int i = 0; i < size; ++i) {
            output[i] = (PixelT)((raw_mag[i] * 255) / max_mag);
        }
    }

    free(raw_mag);
}

// ============================================================
// RVV Optimized Gradient Magnitude (Phase 6) — L1 Norm Only
// Key ideas: abs-value via negate+max trick, vector reduction
// for global max (vredmaxu), two-pass normalize with vdivu.
// ============================================================
#ifdef __riscv
#include <riscv_vector.h>

void gradient_magnitude_rvv(const int16_t* grad_x, const int16_t* grad_y, uint8_t* output, int width, int height) {
    int size = width * height;

    // Temporary 32-bit buffer for raw magnitudes before normalization.
    // Must be 32-bit because L1 max = 2*32767 = 65534, exceeds uint16 range.
    uint32_t* raw_mag = (uint32_t*)aligned_alloc(32, size * sizeof(uint32_t));

    // -------------------------------------------------------
    // PASS 1: Compute L1 Magnitude and Find Global Maximum
    // -------------------------------------------------------
    int vl;

    // __riscv_vsetvlmax_e32m1()
    // WHAT: Returns the maximum possible vl for 32-bit elements with LMUL=1.
    //       This is VLEN/32 elements (e.g., 4 at VLEN=128, 8 at VLEN=256).
    // WHY: We use this to initialize vmax_vec with the right number of elements.
    //      The reduction register must be initialized before any vredmaxu call.
    // VLEN EFFECT: Returns different sizes at different VLEN — handled automatically.
    vuint32m1_t vmax_vec = __riscv_vmv_v_x_u32m1(0, __riscv_vsetvlmax_e32m1());

    for (int i = 0; i < size; i += vl) {

        // __riscv_vsetvl_e16m1(size - i)
        // WHAT: Sets vector length for int16 (e16) elements with LMUL=1.
        //       Returns vl = min(size-i, VLEN/16) — elements remaining to process.
        // WHY e16m1: Gradient arrays are int16_t. LMUL=1 keeps register usage low
        //            while still allowing the widening to m2 for the 32-bit output.
        // VLEN EFFECT: VLEN=128→vl≤8. VLEN=256→vl≤16. VLEN=512→vl≤32.
        vl = __riscv_vsetvl_e16m1(size - i);

        // __riscv_vle16_v_i16m1(ptr, vl)
        // WHAT: Vector Load Elements (16-bit signed) — loads vl consecutive int16
        //       gradient values from memory.
        // WHY i16m1: Gradients are stored as int16_t (SoA layout) so a sequential
        //            load retrieves vl values in a single instruction.
        vint16m1_t vx = __riscv_vle16_v_i16m1(&grad_x[i], vl);
        vint16m1_t vy = __riscv_vle16_v_i16m1(&grad_y[i], vl);

        // Absolute value using the negate-and-max trick.
        // RVV has no dedicated integer abs instruction, so we compute:
        //   abs(x) = max(x, -x)

        // __riscv_vrsub_vx_i16m1(vx, 0, vl)
        // WHAT: Vector Reverse SUBtract — computes 0 - vx element-wise (i.e., negation).
        // WHY vrsub instead of vneg: vrsub_vx with scalar=0 is the idiomatic RVV
        //     negation. Some toolchain versions lack a standalone vneg intrinsic.
        vint16m1_t neg_vx = __riscv_vrsub_vx_i16m1(vx, 0, vl);

        // __riscv_vmax_vv_i16m1(vx, neg_vx, vl)
        // WHAT: Element-wise maximum — picks the larger of vx[i] and neg_vx[i].
        // WHY: max(x, -x) = |x|. This is a branchless absolute value.
        //      For positive x: max(x, -x) = x. For negative x: max(x, -x) = -x.
        vint16m1_t abs_vx = __riscv_vmax_vv_i16m1(vx, neg_vx, vl);

        vint16m1_t neg_vy = __riscv_vrsub_vx_i16m1(vy, 0, vl);
        vint16m1_t abs_vy = __riscv_vmax_vv_i16m1(vy, neg_vy, vl);

        // __riscv_vadd_vv_i16m1(abs_vx, abs_vy, vl)
        // WHAT: Element-wise add — computes L1 norm: |Gx| + |Gy| for each pixel.
        // WHY L1: Avoids sqrt (expensive on RISC-V without FPU vector ops).
        //         L1 slightly overestimates diagonal gradients but is fast and sufficient
        //         for edge detection where we normalize anyway.
        vint16m1_t sum16 = __riscv_vadd_vv_i16m1(abs_vx, abs_vy, vl);

        // __riscv_vreinterpret_v_i16m1_u16m1(sum16)
        // WHAT: Reinterprets int16 bits as uint16. No data change.
        // WHY: vzext (zero-extend) requires unsigned input. Magnitudes are always
        //      non-negative (we just computed abs values), so reinterpret is safe.
        vuint16m1_t usum16 = __riscv_vreinterpret_v_i16m1_u16m1(sum16);

        // __riscv_vzext_vf2_u32m2(usum16, vl)
        // WHAT: Zero-extends each uint16 to uint32, outputting LMUL=2 (m2).
        // WHY: We store raw magnitudes as uint32 to avoid overflow during the
        //      normalization multiply (raw_mag[i] * 255) in Pass 2.
        // CRITICAL LMUL CHAIN: u16m1 → vzext_vf2 → u32m2 (LMUL doubled).
        vuint32m2_t sum32 = __riscv_vzext_vf2_u32m2(usum16, vl);

        // __riscv_vse32_v_u32m2(ptr, sum32, vl)
        // WHAT: Vector Store Elements (32-bit) — writes vl uint32 magnitudes to
        //       the temporary raw_mag buffer.
        // WHY: We store raw values and normalize in Pass 2 because normalization
        //      requires the global maximum, which is only known after the full pass.
        __riscv_vse32_v_u32m2(&raw_mag[i], sum32, vl);

        // __riscv_vredmaxu_vs_u32m2_u32m1(sum32, vmax_vec, vl)
        // WHAT: Vector REDuction MAXimum Unsigned — finds the maximum element
        //       across all vl lanes of sum32 and accumulates it into vmax_vec[0].
        //       This is a HORIZONTAL operation: vl elements → 1 scalar result.
        // WHY: We need the global maximum across the entire image to normalize.
        //      This reduction runs in O(log vl) steps — much faster than a scalar loop.
        //      The result is accumulated across all strip-mining iterations so that
        //      vmax_vec[0] holds the global max after the full loop completes.
        // VLEN EFFECT: At wider VLEN, more elements are reduced per call; result correct.
        vmax_vec = __riscv_vredmaxu_vs_u32m2_u32m1(sum32, vmax_vec, vl);
    }

    // __riscv_vmv_x_s_u32m1_u32(vmax_vec)
    // WHAT: Move Vector to Scalar — extracts element[0] of vmax_vec as a C uint32.
    //       This is how you retrieve a reduction result back into regular C code.
    // WHY: The reduction intrinsic writes its result to element[0] of a vector register.
    //      vmv_x_s bridges the gap between the vector world and scalar C variables.
    uint32_t global_max = __riscv_vmv_x_s_u32m1_u32(vmax_vec);
    if (global_max == 0) global_max = 1; // Guard against all-zero image (divide by zero)

    // -------------------------------------------------------
    // PASS 2: Normalize raw magnitudes to [0, 255]
    // -------------------------------------------------------
    for (int i = 0; i < size; i += vl) {

        // __riscv_vsetvl_e32m4(size - i)
        // WHAT: Sets vector length for uint32 (e32) elements with LMUL=4.
        // WHY e32m4: We read uint32 raw magnitudes here. LMUL=4 allows more elements
        //            per iteration. The narrowing chain (32→16→8) halves LMUL twice,
        //            ending at m1 for the final uint8 store — perfectly balanced.
        // VLEN EFFECT: VLEN=128,m4 → vl≤16. VLEN=256,m4 → vl≤32. VLEN=512,m4 → vl≤64.
        vl = __riscv_vsetvl_e32m4(size - i);

        // __riscv_vle32_v_u32m4(ptr, vl)
        // WHAT: Loads vl uint32 values from the raw_mag buffer.
        // WHY u32m4: Matches the type and LMUL set by vsetvl above.
        vuint32m4_t vmag = __riscv_vle32_v_u32m4(&raw_mag[i], vl);

        // __riscv_vmul_vx_u32m4(vmag, 255, vl)
        // WHAT: Multiplies each uint32 magnitude by scalar 255.
        // WHY: Step 1 of normalization formula: output = (raw_mag * 255) / global_max.
        //      Multiplying first (before dividing) preserves integer precision.
        //      Using 32-bit avoids overflow: max_mag*255 ≤ 65534*255 = 16,711,170 < 2^32.
        vmag = __riscv_vmul_vx_u32m4(vmag, 255, vl);

        // __riscv_vdivu_vx_u32m4(vmag, global_max, vl)
        // WHAT: Unsigned vector divide by scalar global_max — completes normalization.
        // WHY vdivu (unsigned): magnitudes and global_max are both non-negative uint32.
        //     Division maps the magnitude range [0, global_max] to [0, 255].
        //     Note: integer division here is acceptable because we're mapping to 8-bit
        //     output where 1-unit rounding errors are invisible.
        vmag = __riscv_vdivu_vx_u32m4(vmag, global_max, vl);

        // __riscv_vncvt_x_x_w_u16m2(vmag, vl)
        // WHAT: Narrowing Convert — halves uint32 to uint16. LMUL: m4 → m2.
        // WHY: Step 1 of narrowing chain to reach uint8 output.
        //      Values are in [0,255] after normalization, so no data loss occurs.
        // CRITICAL LMUL CHAIN: u32m4 → vncvt → u16m2 (LMUL halved).
        vuint16m2_t vmag16 = __riscv_vncvt_x_x_w_u16m2(vmag, vl);

        // __riscv_vncvt_x_x_w_u8m1(vmag16, vl)
        // WHAT: Second narrowing — halves uint16 to uint8. LMUL: m2 → m1.
        // WHY: Final step produces the 8-bit output pixel for the magnitude image.
        // CRITICAL LMUL CHAIN: u16m2 → vncvt → u8m1 (LMUL halved again).
        //      Full Pass 2 chain: u32m4 → u16m2 → u8m1.
        vuint8m1_t vmag8 = __riscv_vncvt_x_x_w_u8m1(vmag16, vl);

        // __riscv_vse8_v_u8m1(ptr, vmag8, vl)
        // WHAT: Stores vl uint8 normalized magnitude pixels to the output buffer.
        // WHY u8m1: Final output is 8-bit grayscale pixels; LMUL=1 after two narrowings.
        __riscv_vse8_v_u8m1(&output[i], vmag8, vl);
    }

    free(raw_mag);
}
#endif

#endif
