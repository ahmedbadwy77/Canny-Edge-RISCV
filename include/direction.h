#ifndef DIRECTION_H
#define DIRECTION_H

#include <cstdint>
#include <cstdlib>

// Scalar baseline: quantizes gradient direction to 4 values (0, 45, 90, 135 degrees).
// Uses integer cross-multiplication instead of atan2().
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

#if defined(__riscv_vector)
#include <riscv_vector.h>
#include <cstdint>

void gradient_direction_rvv(const int16_t* grad_x,
                            const int16_t* grad_y,
                            uint8_t* dir,
                            int width, int height) {
    int size = width * height;

    for (int i = 0; i < size; ) {
        // vsetvl selects the active element count for the remaining gradients.
        // e16m2 matches int16 gradient inputs; larger VLEN only increases vl.
        size_t vl = __riscv_vsetvl_e16m2(size - i);

        // Load Gx/Gy as signed 16-bit vectors. LMUL=m2 balances throughput
        // with register pressure; VLEN changes the number of loaded pixels.
        vint16m2_t vdx = __riscv_vle16_v_i16m2(&grad_x[i], vl);
        vint16m2_t vdy = __riscv_vle16_v_i16m2(&grad_y[i], vl);

        // Broadcast zero in i16m2 for sign and zero comparisons.
        vint16m2_t vzero16 = __riscv_vmv_v_x_i16m2(0, vl);

        // Vector comparisons create masks for sign/zero tests. The mask type
        // is b8 because i16m2 has the same element grouping; VLEN only affects
        // how many mask bits are active.
        vbool8_t dx_neg_mask = __riscv_vmslt_vv_i16m2_b8(vdx, vzero16, vl);
        vbool8_t dy_neg_mask = __riscv_vmslt_vv_i16m2_b8(vdy, vzero16, vl);
        vbool8_t dx_pos_mask = __riscv_vmsgt_vv_i16m2_b8(vdx, vzero16, vl);
        vbool8_t dy_pos_mask = __riscv_vmsgt_vv_i16m2_b8(vdy, vzero16, vl);
        vbool8_t dx_zero_mask = __riscv_vmseq_vv_i16m2_b8(vdx, vzero16, vl);
        vbool8_t dy_zero_mask = __riscv_vmseq_vv_i16m2_b8(vdy, vzero16, vl);

        // Mask logical AND finds pixels where both gradients are zero.
        vbool8_t both_zero_mask = __riscv_vmand_mm_b8(dx_zero_mask, dy_zero_mask, vl);

        // vneg computes -dx/-dy and vmerge selects the negated value only for
        // negative lanes, producing abs(dx)/abs(dy). LMUL remains m2.
        vint16m2_t abs_dx = __riscv_vmerge_vvm_i16m2(vdx, __riscv_vneg_v_i16m2(vdx, vl), dx_neg_mask, vl);
        vint16m2_t abs_dy = __riscv_vmerge_vvm_i16m2(vdy, __riscv_vneg_v_i16m2(vdy, vl), dy_neg_mask, vl);

        // Widening multiply promotes i16m2 to i32m4 to avoid overflow in the
        // integer angle thresholds. Widening doubles LMUL; VLEN still only
        // changes vl, not the threshold equations.
        vint32m4_t dy_times_1000 = __riscv_vwmul_vx_i32m4(abs_dy, 1000, vl);
        vint32m4_t dx_times_414 = __riscv_vwmul_vx_i32m4(abs_dx, 414, vl);
        vint32m4_t dx_times_2414 = __riscv_vwmul_vx_i32m4(abs_dx, 2414, vl);

        // Compare widened threshold products to classify horizontal and
        // vertical directions. The b8 mask matches the widened i32m4 grouping.
        vbool8_t horizontal_mask = __riscv_vmsle_vv_i32m4_b8(dy_times_1000, dx_times_414, vl);
        vbool8_t vertical_mask = __riscv_vmsgt_vv_i32m4_b8(dy_times_1000, dx_times_2414, vl);

        // Combine sign masks: same sign maps diagonal gradients to 45 degrees,
        // opposite sign maps them to 135 degrees.
        vbool8_t both_pos_mask = __riscv_vmand_mm_b8(dx_pos_mask, dy_pos_mask, vl);
        vbool8_t both_neg_mask = __riscv_vmand_mm_b8(dx_neg_mask, dy_neg_mask, vl);
        vbool8_t same_sign_mask = __riscv_vmor_mm_b8(both_pos_mask, both_neg_mask, vl);

        // Build the output direction vector with masked merges. u8m1 is enough
        // for 0/45/90/135 degree labels; different VLEN values only change how
        // many labels are produced per strip-mined iteration.
        vuint8m1_t vdir = __riscv_vmv_v_x_u8m1(135, vl);
        vdir = __riscv_vmerge_vvm_u8m1(vdir, __riscv_vmv_v_x_u8m1(45, vl), same_sign_mask, vl);
        vdir = __riscv_vmerge_vvm_u8m1(vdir, __riscv_vmv_v_x_u8m1(0, vl), horizontal_mask, vl);
        vdir = __riscv_vmerge_vvm_u8m1(vdir, __riscv_vmv_v_x_u8m1(90, vl), vertical_mask, vl);
        vdir = __riscv_vmerge_vvm_u8m1(vdir, __riscv_vmv_v_x_u8m1(0, vl), both_zero_mask, vl);

        // Store vl byte-sized direction labels. u8m1 keeps the output compact.
        __riscv_vse8_v_u8m1(&dir[i], vdir, vl);
        i += vl;
    }
}

#endif

#endif
