#ifndef GAUSSIAN_H
#define GAUSSIAN_H

#include <cstdint>

// 5x5 Filter Matrix (sum = 273)
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

// ==========================================
// RVV Optimization: Gaussian Blur (Phase 6)
// ==========================================
#ifdef __riscv
#include <riscv_vector.h>

void gaussian_blur_rvv(const uint8_t* input, uint8_t* output, int width, int height) {
    // تصفير الصورة الأول عشان الأطراف اللي هنتجاهلها تبقى سودة
    for(int i = 0; i < width * height; i++) output[i] = 0;

    // نصيحة الدليل: معالجة الأجزاء الداخلية فقط وتجاهل الأطراف في البداية
    for (int y = 2; y < height - 2; ++y) {
        int vl;
        // Strip-mining على عرض الصورة (ناقص الأطراف)
        for (int x = 2; x < width - 2; x += vl) {
            vl = __riscv_vsetvl_e8m1(width - 2 - x);

            // مسجل 32-bit (LMUL=4) لتجميع ناتج الضرب بصفر
            vint32m4_t vsum = __riscv_vmv_v_x_i32m4(0, vl);

            // المرور على المصفوفة 5x5
            for (int ky = -2; ky <= 2; ++ky) {
                for (int kx = -2; kx <= 2; ++kx) {
                    // 1. تحميل البكسلات (8-bit)
                    vuint8m1_t vpix = __riscv_vle8_v_u8m1(&input[(y + ky) * width + (x + kx)], vl);
                    
                    // 2. توسيع البكسل لـ 16-bit
                    vuint16m2_t vpix_u16 = __riscv_vzext_vf2_u16m2(vpix, vl);
                    // تحويله لـ Signed عشان ينفع يتضرب في الـ Kernel
                    vint16m2_t vpix_i16 = __riscv_vreinterpret_v_u16m2_i16m2(vpix_u16);

                    // 3. توسيع وضرب في المعامل (16-bit * 16-bit = 32-bit)
                    int16_t coeff = GAUSSIAN_KERNEL[ky + 2][kx + 2];
                    vint32m4_t vprod = __riscv_vwmul_vx_i32m4(vpix_i16, coeff, vl);

                    // 4. التجميع
                    vsum = __riscv_vadd_vv_i32m4(vsum, vprod, vl);
                }
            }

            // 5. خدعة القسمة السريعة: (sum * 240) >> 16
            vsum = __riscv_vmul_vx_i32m4(vsum, 240, vl);
            vsum = __riscv_vsra_vx_i32m4(vsum, 16, vl);

            // 6. التأكد إن القيم مابين 0 و 255
            vsum = __riscv_vmax_vx_i32m4(vsum, 0, vl);
            vsum = __riscv_vmin_vx_i32m4(vsum, 255, vl);

            // 7. تصغير الحجم للرجوع لـ 8-bit
            vuint32m4_t vsum_u32 = __riscv_vreinterpret_v_i32m4_u32m4(vsum);
            vuint16m2_t vsum_u16 = __riscv_vncvt_x_x_w_u16m2(vsum_u32, vl);
            vuint8m1_t vsum_u8 = __riscv_vncvt_x_x_w_u8m1(vsum_u16, vl);

            // 8. حفظ البكسلات في الصورة الناتجة
            __riscv_vse8_v_u8m1(&output[y * width + x], vsum_u8, vl);
        }
    }
}
#endif

#endif