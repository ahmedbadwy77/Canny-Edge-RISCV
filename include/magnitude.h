#ifndef MAGNITUDE_H
#define MAGNITUDE_H

#include <cstdint>
#include <cmath>
#include <algorithm>
#include <cstdlib>

// Enum for magnitude method
enum class MagMethod {
    L1,
    L2
};

template<typename GradT = int16_t, typename PixelT = uint8_t>
void gradient_magnitude_scalar(const GradT* grad_x, const GradT* grad_y, PixelT* output, int width, int height, MagMethod method) {
    int size = width * height;
    
    // Allocate temporary buffer for raw magnitude (uint32_t to avoid overflow)
    uint32_t* raw_mag = (uint32_t*)aligned_alloc(32, size * sizeof(uint32_t));
    uint32_t max_mag = 0;

    // Pass 1: Calculate magnitude and find max value
    for (int i = 0; i < size; ++i) {
        uint32_t mag = 0;
        if (method == MagMethod::L1) {
            mag = std::abs(grad_x[i]) + std::abs(grad_y[i]);
        } else {
            // L2: sqrt(Gx^2 + Gy^2)
            mag = (uint32_t)std::round(std::sqrt(grad_x[i] * grad_x[i] + grad_y[i] * grad_y[i]));
        }
        
        raw_mag[i] = mag;
        
        if (mag > max_mag) {
            max_mag = mag;
        }
    }

    // Pass 2: Normalize to [0, 255]
    if (max_mag == 0) {
        for (int i = 0; i < size; ++i) {
            output[i] = 0;
        }
    } else {
        for (int i = 0; i < size; ++i) {
            output[i] = (PixelT)((raw_mag[i] * 255) / max_mag);
        }
    }

    free(raw_mag);
}

// ==========================================
// RVV Optimization: Gradient Magnitude (Phase 6)
// ==========================================
#ifdef __riscv
#include <riscv_vector.h>

void gradient_magnitude_rvv(const int16_t* grad_x, const int16_t* grad_y, uint8_t* output, int width, int height) {
    int size = width * height;
    // حجز ميموري مؤقتة للـ Magnitude الخام (32-bit عشان مفيش حاجة تضرب مننا)
    uint32_t* raw_mag = (uint32_t*)aligned_alloc(32, size * sizeof(uint32_t));

    // ----------------------------------------------------
    // Pass 1: L1 Magnitude (|Gx| + |Gy|) & Find Max
    // ----------------------------------------------------
    int vl;
    uint32_t global_max = 0;
    
    // تجهيز مسجل Vector من نوع Scalar (عنصر واحد) بصفر عشان نخزن فيه الـ Max
    vuint32m1_t vmax_vec = __riscv_vmv_v_x_u32m1(0, __riscv_vsetvlmax_e32m1());

    for (int i = 0; i < size; i += vl) {
        // Strip-mining: تحديد طول الفيكتور على حسب الباقي من الصورة
        vl = __riscv_vsetvl_e16m1(size - i);

        // 1. تحميل Gx و Gy (كـ int16_t)
        vint16m1_t vx = __riscv_vle16_v_i16m1(&grad_x[i], vl);
        vint16m1_t vy = __riscv_vle16_v_i16m1(&grad_y[i], vl);

        // 2. حساب القيمة المطلقة: لا توجد دالة abs مباشرة، فبنحسب السالب وناخد الأكبر
        vint16m1_t neg_vx = __riscv_vrsub_vx_i16m1(vx, 0, vl); // 0 - vx
        vint16m1_t abs_vx = __riscv_vmax_vv_i16m1(vx, neg_vx, vl); // max(vx, -vx)

        vint16m1_t neg_vy = __riscv_vrsub_vx_i16m1(vy, 0, vl);
        vint16m1_t abs_vy = __riscv_vmax_vv_i16m1(vy, neg_vy, vl);

        // 3. الجمع L1 Norm (|Gx| + |Gy|)
        vint16m1_t sum16 = __riscv_vadd_vv_i16m1(abs_vx, abs_vy, vl);

        // 4. توسيع الحجم (Widening) من 16-bit لـ 32-bit
        // ده بيضاعف الـ LMUL من m1 لـ m2
        vuint16m1_t usum16 = __riscv_vreinterpret_v_i16m1_u16m1(sum16);
        vuint32m2_t sum32 = __riscv_vzext_vf2_u32m2(usum16, vl);

        // 5. حفظ النتيجة الخام في الميموري
        __riscv_vse32_v_u32m2(&raw_mag[i], sum32, vl);

        // 6. الـ Reduction السحري: إيجاد أكبر رقم في الفيكتور الحالي وحفظه
        vmax_vec = __riscv_vredmaxu_vs_u32m2_u32m1(sum32, vmax_vec, vl);
    }

    // استخراج قيمة الـ Max النهائية للـ C++ العادي
    global_max = __riscv_vmv_x_s_u32m1_u32(vmax_vec);
    if (global_max == 0) global_max = 1;

    // ----------------------------------------------------
    // Pass 2: Normalization [0, 255]
    // ----------------------------------------------------
    for (int i = 0; i < size; i += vl) {
        // نستخدم e32m4 عشان هنقرأ 32-bit وننزل لـ 8-bit
        vl = __riscv_vsetvl_e32m4(size - i);

        // تحميل الـ Magnitude الخام
        vuint32m4_t vmag = __riscv_vle32_v_u32m4(&raw_mag[i], vl);

        // (mag * 255) / max
        vmag = __riscv_vmul_vx_u32m4(vmag, 255, vl);
        vmag = __riscv_vdivu_vx_u32m4(vmag, global_max, vl);

        // تضييق الحجم (Narrowing) من 32-bit لـ 16-bit
        vuint16m2_t vmag16 = __riscv_vncvt_x_x_w_u16m2(vmag, vl);
        // تضييق الحجم من 16-bit لـ 8-bit
        vuint8m1_t vmag8 = __riscv_vncvt_x_x_w_u8m1(vmag16, vl);

        // الحفظ النهائي كبكسل 8-bit
        __riscv_vse8_v_u8m1(&output[i], vmag8, vl);
    }

    free(raw_mag);
}
#endif

#endif
