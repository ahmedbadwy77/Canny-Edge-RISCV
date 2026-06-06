#ifndef DIRECTION_H
#define DIRECTION_H

#include <cstdint>
#include <cstdlib>

template<typename GradT = int16_t>
void gradient_direction_scalar(const GradT* grad_x, const GradT* grad_y, uint8_t* dir_output, int width, int height) {
    int size = width * height;
    
    for (int i = 0; i < size; ++i) {
        GradT dx = grad_x[i];
        GradT dy = grad_y[i];

        if (dx == 0 && dy == 0) {
            dir_output[i] = 0; // Default to 0 degrees
            continue;
        }

        int32_t abs_dx = std::abs(dx);
        int32_t abs_dy = std::abs(dy);

        // Integer approximation for angle bounds to avoid atan2()
        // tan(22.5) ~= 0.414 -> abs_dy * 1000 <= abs_dx * 414
        // tan(67.5) ~= 2.414 -> abs_dy * 1000 >  abs_dx * 2414

        if (abs_dy * 1000 <= abs_dx * 414) {
            dir_output[i] = 0;   // Horizontal
        } else if (abs_dy * 1000 > abs_dx * 2414) {
            dir_output[i] = 90;  // Vertical
        } else {
            // Diagonal
            if ((dx > 0 && dy > 0) || (dx < 0 && dy < 0)) {
                dir_output[i] = 45;  // Positive diagonal
            } else {
                dir_output[i] = 135; // Negative diagonal
            }
        }
    }
}
#ifdef __riscv
#include <riscv_vector.h>
#include <cstdint>

void gradient_direction_rvv(const int16_t* grad_x,
                            const int16_t* grad_y,
                            uint8_t* dir,
                            int width, int height) {
    for (int i = 0; i < width * height; i += 8) {
        size_t vl = __riscv_vsetvl_e16m2(width * height - i);

        // Load gradients
        vint16m2_t vdx = __riscv_vle16_v_i16m2(&grad_x[i], vl);
        vint16m2_t vdy = __riscv_vle16_v_i16m2(&grad_y[i], vl);

        // Absolute value of vdx
	vbool8_t neg_mask = __riscv_vmslt_vv_i16m2_b8(vdx, __riscv_vmv_v_x_i16m2(0, vl), vl);
	vint16m2_t vdx_neg = __riscv_vneg_v_i16m2(vdx, vl);
	vint16m2_t vabs_dx = __riscv_vmerge_vvm_i16m2(vdx, vdx_neg, neg_mask, vl);

	// Absolute value of vdy
	vbool8_t neg_mask_y = __riscv_vmslt_vv_i16m2_b8(vdy, __riscv_vmv_v_x_i16m2(0, vl), vl);
	vint16m2_t vdy_neg = __riscv_vneg_v_i16m2(vdy, vl);
	vint16m2_t vabs_dy = __riscv_vmerge_vvm_i16m2(vdy, vdy_neg, neg_mask_y, vl);

	// Direction classification
	vbool8_t horiz_mask = __riscv_vmsgt_vv_i16m2_b8(vabs_dx, vabs_dy, vl);
	vbool8_t vert_mask  = __riscv_vmsgt_vv_i16m2_b8(vabs_dy, vabs_dx, vl);


        vuint8m1_t vdir = __riscv_vmv_v_x_u8m1(45, vl); // default diagonal
        vdir = __riscv_vmerge_vvm_u8m1(vdir, __riscv_vmv_v_x_u8m1(0, vl), horiz_mask, vl);
        vdir = __riscv_vmerge_vvm_u8m1(vdir, __riscv_vmv_v_x_u8m1(90, vl), vert_mask, vl);

        // Store result
        __riscv_vse8_v_u8m1(&dir[i], vdir, vl);
    }
}

#endif

#endif
