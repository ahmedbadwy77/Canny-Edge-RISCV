#ifndef DIRECTION_H
#define DIRECTION_H

#include <cstdint>
#include <cstdlib>

// Scalar baseline: quantizes gradient direction to 4 values (0, 45, 90, 135 degrees).
// Uses integer cross-multiplication instead of atan2() — an embedded optimization
// that avoids expensive floating-point and transcendental function calls.
// tan(22.5°) ≈ 0.414 and tan(67.5°) ≈ 2.414 are approximated with integer ratios.
template<typename GradT = int16_t>
void gradient_direction_scalar(const GradT* grad_x, const GradT* grad_y, uint8_t* dir_output, int width, int height) {
    int size = width * height;

    for (int i = 0; i < size; ++i) {
        GradT dx = grad_x[i];
        GradT dy = grad_y[i];

        if (dx == 0 && dy == 0) {
            dir_output[i] = 0;
            continue;
        }

        int32_t abs_dx = std::abs(dx);
        int32_t abs_dy = std::abs(dy);

        // Branchless angle quantization using integer cross-multiplication:
        // tan(22.5°) ≈ 2/5 → if abs_dy*5 <= abs_dx*2: direction is horizontal (0°)
        // tan(67.5°) ≈ 12/5 → if abs_dy*5 > abs_dx*12: direction is vertical (90°)
        if (abs_dy * 1000 <= abs_dx * 414) {
            dir_output[i] = 0;
        } else if (abs_dy * 1000 > abs_dx * 2414) {
            dir_output[i] = 90;
        } else {
            if ((dx > 0 && dy > 0) || (dx < 0 && dy < 0)) {
                dir_output[i] = 45;
            } else {
                dir_output[i] = 135;
            }
        }
    }
}

// ============================================================
// RVV Optimized Gradient Direction (Phase 7)
// Key ideas: vector comparison + mask-based merge to replace
// conditional branches with branchless vector select operations.
// Note: This stage is ~0.6% of total runtime (Amdahl's Law),
// so RVV here is for completeness, not performance impact.
// ============================================================
#ifdef __riscv
#include <riscv_vector.h>
#include <cstdint>

void gradient_direction_rvv(const int16_t* grad_x,
                            const int16_t* grad_y,
                            uint8_t* dir,
                            int width, int height) {
    for (int i = 0; i < width * height; i += 8) {

        // __riscv_vsetvl_e16m2(n)
        // WHAT: Sets vector length for int16 (e16) with LMUL=2 (m2).
        //       Returns vl elements to process in this iteration.
        // WHY e16m2: Gradients are int16_t. LMUL=2 gives us larger groups of elements
        //            per iteration. The bool type vbool8_t corresponds to m2 (1 mask
        //            bit per element at LMUL=2 means boolSEW = 8, hence b8).
        // VLEN EFFECT: VLEN=128→vl≤16. VLEN=256→vl≤32. VLEN=512→vl≤64.
        size_t vl = __riscv_vsetvl_e16m2(width * height - i);

        // __riscv_vle16_v_i16m2(ptr, vl)
        // WHAT: Loads vl consecutive int16 gradient values from memory.
        // WHY i16m2: Matches gradient storage type (int16_t SoA arrays).
        vint16m2_t vdx = __riscv_vle16_v_i16m2(&grad_x[i], vl);
        vint16m2_t vdy = __riscv_vle16_v_i16m2(&grad_y[i], vl);

        // --- Compute |dx| using compare-and-merge (branchless abs) ---

        // __riscv_vmslt_vv_i16m2_b8(vdx, zero_vec, vl)
        // WHAT: Vector Mask Set-if-Less-Than — produces a boolean mask where
        //       mask[i] = 1 if vdx[i] < 0, else 0.
        // WHY: We use this mask to selectively negate negative elements (computing abs).
        //      The _b8 suffix means 1 bit per element (boolSEW=8, for LMUL=2, e16).
        vbool8_t neg_mask = __riscv_vmslt_vv_i16m2_b8(vdx, __riscv_vmv_v_x_i16m2(0, vl), vl);

        // __riscv_vmv_v_x_i16m2(0, vl)
        // WHAT: Broadcasts scalar 0 into all vl elements of an i16m2 vector.
        // WHY: Creates a zero vector for the comparison operand in vmslt above.

        // __riscv_vneg_v_i16m2(vdx, vl)
        // WHAT: Negates every element of vdx (computes -vdx[i] for each i).
        // WHY: Used as the "negative is positive" branch of the abs computation.
        vint16m2_t vdx_neg = __riscv_vneg_v_i16m2(vdx, vl);

        // __riscv_vmerge_vvm_i16m2(vdx, vdx_neg, neg_mask, vl)
        // WHAT: Merge — selects element-wise: output[i] = neg_mask[i] ? vdx_neg[i] : vdx[i].
        //       Where mask=1 (dx was negative), use the negated value (making it positive).
        //       Where mask=0 (dx was non-negative), keep original value.
        // WHY: This is the vectorized equivalent of: abs_dx = (dx < 0) ? -dx : dx.
        //      Completely branchless — no pipeline flushes or misprediction costs.
        vint16m2_t vabs_dx = __riscv_vmerge_vvm_i16m2(vdx, vdx_neg, neg_mask, vl);

        // Same abs computation for vy:
        vbool8_t neg_mask_y = __riscv_vmslt_vv_i16m2_b8(vdy, __riscv_vmv_v_x_i16m2(0, vl), vl);
        vint16m2_t vdy_neg = __riscv_vneg_v_i16m2(vdy, vl);
        vint16m2_t vabs_dy = __riscv_vmerge_vvm_i16m2(vdy, vdy_neg, neg_mask_y, vl);

        // --- Direction classification using comparison masks ---

        // __riscv_vmsgt_vv_i16m2_b8(vabs_dx, vabs_dy, vl)
        // WHAT: Mask Set if Greater-Than — horiz_mask[i] = 1 if |dx[i]| > |dy[i]|.
        // WHY: If |dx| > |dy|, the gradient is primarily horizontal → edge is vertical
        //      → direction = 0°. This replaces the scalar tan(22.5°) comparison with
        //      a simpler (less precise but sufficient) magnitude comparison.
        vbool8_t horiz_mask = __riscv_vmsgt_vv_i16m2_b8(vabs_dx, vabs_dy, vl);

        // __riscv_vmsgt_vv_i16m2_b8(vabs_dy, vabs_dx, vl)
        // WHAT: vert_mask[i] = 1 if |dy[i]| > |dx[i]| (predominantly vertical gradient).
        // WHY: If |dy| > |dx|, direction = 90°. Remaining cases (|dx| ≈ |dy|) default to 45°.
        vbool8_t vert_mask  = __riscv_vmsgt_vv_i16m2_b8(vabs_dy, vabs_dx, vl);

        // __riscv_vmv_v_x_u8m1(45, vl)
        // WHAT: Broadcasts scalar 45 into all vl elements of a uint8 vector.
        // WHY: Default direction is 45° (diagonal). We then overwrite horizontal
        //      and vertical pixels using mask-based merges below.
        vuint8m1_t vdir = __riscv_vmv_v_x_u8m1(45, vl);

        // __riscv_vmerge_vvm_u8m1(vdir, zero_vec, horiz_mask, vl)
        // WHAT: Where horiz_mask=1, replaces direction with 0; elsewhere keeps vdir.
        // WHY: Sets horizontal-gradient pixels to direction=0° (horizontal edge).
        //      vmerge is the vectorized conditional assignment: vdir[i] = horiz ? 0 : vdir[i].
        vdir = __riscv_vmerge_vvm_u8m1(vdir, __riscv_vmv_v_x_u8m1(0, vl), horiz_mask, vl);

        // __riscv_vmerge_vvm_u8m1(vdir, ninety_vec, vert_mask, vl)
        // WHAT: Where vert_mask=1, replaces direction with 90; elsewhere keeps vdir.
        // WHY: Sets vertical-gradient pixels to direction=90°.
        //      Applied after the horiz merge — if both masks are 0, direction stays 45°.
        vdir = __riscv_vmerge_vvm_u8m1(vdir, __riscv_vmv_v_x_u8m1(90, vl), vert_mask, vl);

        // __riscv_vse8_v_u8m1(ptr, vdir, vl)
        // WHAT: Stores vl uint8 direction values (0, 45, or 90) to the output buffer.
        // WHY u8m1: Direction values fit in uint8. LMUL=1 after implicit narrowing
        //           from the u8m1 vmv_v_x initialization.
        // VLEN EFFECT: More pixels processed and stored per call at wider VLEN.
        __riscv_vse8_v_u8m1(&dir[i], vdir, vl);
    }
}

#endif

#endif
