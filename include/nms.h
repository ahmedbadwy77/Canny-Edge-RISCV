#ifndef NMS_H
#define NMS_H

#include <cstdint>
#if defined(__riscv_vector)
#include <riscv_vector.h>
#endif

inline void non_max_suppression(const uint8_t* magnitude,
                                const uint8_t* direction,
                                uint8_t* output,
                                int width,
                                int height) {
    // Zero the image border because NMS compares each interior pixel with
    // neighbors on both sides of the local gradient direction.
    for (int x = 0; x < width; x++) {
        output[x] = 0;
        output[(height - 1) * width + x] = 0;
    }
    for (int y = 0; y < height; y++) {
        output[y * width] = 0;
        output[y * width + width - 1] = 0;
    }

#if defined(__riscv_vector)
    for (int y = 1; y < height - 1; y++) {
        int vl;
        for (int x = 1; x < width - 1; x += vl) {
            // vsetvl selects a tail-safe vector length for this row. e8m1
            // matches byte magnitude/direction buffers; larger VLEN only
            // increases vl and reduces loop iterations.
            vl = __riscv_vsetvl_e8m1(width - 1 - x);
            int idx = y * width + x;

            // Load current magnitude and quantized direction. LMUL=m1 is
            // enough for byte data and keeps register pressure low while all
            // eight neighbor vectors are live.
            vuint8m1_t vmag = __riscv_vle8_v_u8m1(&magnitude[idx], vl);
            vuint8m1_t vdir = __riscv_vle8_v_u8m1(&direction[idx], vl);

            // Load the eight neighboring magnitude vectors used by the four
            // quantized direction cases. Each load covers vl adjacent pixels.
            vuint8m1_t left   = __riscv_vle8_v_u8m1(&magnitude[idx - 1], vl);
            vuint8m1_t right  = __riscv_vle8_v_u8m1(&magnitude[idx + 1], vl);
            vuint8m1_t top    = __riscv_vle8_v_u8m1(&magnitude[idx - width], vl);
            vuint8m1_t bottom = __riscv_vle8_v_u8m1(&magnitude[idx + width], vl);
            vuint8m1_t tr     = __riscv_vle8_v_u8m1(&magnitude[idx - width + 1], vl);
            vuint8m1_t bl     = __riscv_vle8_v_u8m1(&magnitude[idx + width - 1], vl);
            vuint8m1_t tl     = __riscv_vle8_v_u8m1(&magnitude[idx - width - 1], vl);
            vuint8m1_t br     = __riscv_vle8_v_u8m1(&magnitude[idx + width + 1], vl);

            // Default direction is 0 degrees, so compare left and right.
            vuint8m1_t n1 = left;
            vuint8m1_t n2 = right;

            // vmseq builds a mask for 90-degree lanes; vmerge selects top and
            // bottom neighbors only for those lanes. LMUL stays m1.
            vbool8_t m90 = __riscv_vmseq_vx_u8m1_b8(vdir, 90, vl);
            n1 = __riscv_vmerge_vvm_u8m1(n1, top, m90, vl);
            n2 = __riscv_vmerge_vvm_u8m1(n2, bottom, m90, vl);

            // Direction 45 degrees compares top-right and bottom-left.
            vbool8_t m45 = __riscv_vmseq_vx_u8m1_b8(vdir, 45, vl);
            n1 = __riscv_vmerge_vvm_u8m1(n1, tr, m45, vl);
            n2 = __riscv_vmerge_vvm_u8m1(n2, bl, m45, vl);

            // Direction 135 degrees compares top-left and bottom-right.
            vbool8_t m135 = __riscv_vmseq_vx_u8m1_b8(vdir, 135, vl);
            n1 = __riscv_vmerge_vvm_u8m1(n1, tl, m135, vl);
            n2 = __riscv_vmerge_vvm_u8m1(n2, br, m135, vl);

            // Unsigned vector comparisons keep a pixel only if its magnitude is
            // at least both selected neighbors. b8 masks match u8m1 data.
            vbool8_t mask1 = __riscv_vmsgeu_vv_u8m1_b8(vmag, n1, vl);
            vbool8_t mask2 = __riscv_vmsgeu_vv_u8m1_b8(vmag, n2, vl);
            vbool8_t final_mask = __riscv_vmand_mm_b8(mask1, mask2, vl);

            // Merge magnitude with zero: active lanes keep vmag, suppressed
            // lanes become 0. Larger VLEN only means more lanes are handled
            // per strip-mined iteration.
            vuint8m1_t zeros = __riscv_vmv_v_x_u8m1(0, vl);
            vuint8m1_t vout = __riscv_vmerge_vvm_u8m1(zeros, vmag, final_mask, vl);

            // Store vl suppressed magnitudes back to the output image.
            __riscv_vse8_v_u8m1(&output[idx], vout, vl);
        }
    }
#else
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            int idx = y * width + x;
            uint8_t mag = magnitude[idx];
            uint8_t dir = direction[idx];
            uint8_t n1, n2;

            if (dir == 0) {
                n1 = magnitude[idx - 1];
                n2 = magnitude[idx + 1];
            } else if (dir == 90) {
                n1 = magnitude[idx - width];
                n2 = magnitude[idx + width];
            } else if (dir == 45) {
                n1 = magnitude[idx - width + 1];
                n2 = magnitude[idx + width - 1];
            } else {
                n1 = magnitude[idx - width - 1];
                n2 = magnitude[idx + width + 1];
            }

            output[idx] = (mag >= n1 && mag >= n2) ? mag : 0;
        }
    }
#endif
}

#endif
