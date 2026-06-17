#include <gtest/gtest.h>
#include <cstdint>
#include <cmath>
#include <cstdlib>

#include "../include/gaussian.h"
#include "../include/sobel.h"
#include "../include/magnitude.h"
#include "../include/direction.h"
#include "../include/nms.h"
#include "../include/threshold.h"
#include "../include/hysteresis.h"

// =============================================================
// Helpers
// =============================================================

static uint8_t* alloc_img(int w, int h, uint8_t fill = 0) {
    uint8_t* p = (uint8_t*)aligned_alloc(32, w * h);
    for (int i = 0; i < w * h; i++) p[i] = fill;
    return p;
}

static int16_t* alloc_grad(int w, int h) {
    int16_t* p = (int16_t*)aligned_alloc(32, w * h * sizeof(int16_t));
    for (int i = 0; i < w * h; i++) p[i] = 0;
    return p;
}

// =============================================================
// STAGE 1 - GAUSSIAN BLUR
// =============================================================

TEST(GaussianBlur, UniformImageStaysConstant) {
    const int W = 64, H = 64;
    uint8_t* in  = alloc_img(W, H, 128);
    uint8_t* out = alloc_img(W, H, 0);
    gaussian_blur_scalar(in, out, W, H);
    // With zero-padding, only pixels with a complete 5x5 neighborhood stay
    // exactly constant. Border pixels are expected to be darker.
    for (int y = 2; y < H - 2; y++)
        for (int x = 2; x < W - 2; x++)
            EXPECT_EQ(out[y * W + x], 128) << "Failed at pixel " << (y * W + x);
    free(in); free(out);
}

TEST(GaussianBlur, ZeroImageStaysZero) {
    const int W = 32, H = 32;
    uint8_t* in  = alloc_img(W, H, 0);
    uint8_t* out = alloc_img(W, H, 0);
    gaussian_blur_scalar(in, out, W, H);
    for (int i = 0; i < W * H; i++)
        EXPECT_EQ(out[i], 0);
    free(in); free(out);
}

TEST(GaussianBlur, OutputAlwaysClamped) {
    const int W = 64, H = 64;
    uint8_t* in  = alloc_img(W, H, 255);
    uint8_t* out = alloc_img(W, H, 0);
    gaussian_blur_scalar(in, out, W, H);
    for (int i = 0; i < W * H; i++) {
        EXPECT_GE(out[i], 0);
        EXPECT_LE(out[i], 255);
    }
    free(in); free(out);
}

TEST(GaussianBlur, ImpulseIsSmoothed) {
    const int W = 32, H = 32;
    uint8_t* in  = alloc_img(W, H, 0);
    uint8_t* out = alloc_img(W, H, 0);
    in[16 * W + 16] = 255;
    gaussian_blur_scalar(in, out, W, H);
    EXPECT_LT(out[16 * W + 16], 255);
    EXPECT_GT(out[16 * W + 16], 0);
    EXPECT_GT(out[16 * W + 17], 0);
    free(in); free(out);
}

// =============================================================
// STAGE 2 - SOBEL GRADIENTS
// =============================================================

TEST(SobelGradients, UniformImageZeroGradient) {
    const int W = 64, H = 64;
    uint8_t* in  = alloc_img(W, H, 100);
    int16_t* gx  = alloc_grad(W, H);
    int16_t* gy  = alloc_grad(W, H);
    sobel_gradients_scalar(in, gx, gy, W, H);
    for (int y = 1; y < H - 1; y++)
        for (int x = 1; x < W - 1; x++) {
            EXPECT_EQ(gx[y * W + x], 0);
            EXPECT_EQ(gy[y * W + x], 0);
        }
    free(in); free(gx); free(gy);
}

TEST(SobelGradients, VerticalEdgeProducesStrongGx) {
    const int W = 64, H = 64;
    uint8_t* in  = alloc_img(W, H, 0);
    int16_t* gx  = alloc_grad(W, H);
    int16_t* gy  = alloc_grad(W, H);
    for (int y = 0; y < H; y++)
        for (int x = W / 2; x < W; x++)
            in[y * W + x] = 255;
    sobel_gradients_scalar(in, gx, gy, W, H);
    for (int y = 1; y < H - 1; y++)
        EXPECT_GT(std::abs(gx[y * W + W / 2]), 0);
    free(in); free(gx); free(gy);
}

TEST(SobelGradients, HorizontalEdgeProducesStrongGy) {
    const int W = 64, H = 64;
    uint8_t* in  = alloc_img(W, H, 0);
    int16_t* gx  = alloc_grad(W, H);
    int16_t* gy  = alloc_grad(W, H);
    for (int y = H / 2; y < H; y++)
        for (int x = 0; x < W; x++)
            in[y * W + x] = 255;
    sobel_gradients_scalar(in, gx, gy, W, H);
    for (int x = 1; x < W - 1; x++)
        EXPECT_GT(std::abs(gy[H / 2 * W + x]), 0);
    free(in); free(gx); free(gy);
}

TEST(SobelGradients, ZeroImageZeroGradient) {
    const int W = 32, H = 32;
    uint8_t* in  = alloc_img(W, H, 0);
    int16_t* gx  = alloc_grad(W, H);
    int16_t* gy  = alloc_grad(W, H);
    sobel_gradients_scalar(in, gx, gy, W, H);
    for (int i = 0; i < W * H; i++) {
        EXPECT_EQ(gx[i], 0);
        EXPECT_EQ(gy[i], 0);
    }
    free(in); free(gx); free(gy);
}

// =============================================================
// STAGE 2b - GRADIENT MAGNITUDE
// =============================================================

TEST(GradientMagnitude, ZeroGradZeroMagnitude_L1) {
    const int W = 32, H = 32;
    int16_t* gx  = alloc_grad(W, H);
    int16_t* gy  = alloc_grad(W, H);
    uint8_t* out = alloc_img(W, H, 0);
    gradient_magnitude_scalar(gx, gy, out, W, H, MagMethod::L1);
    for (int i = 0; i < W * H; i++)
        EXPECT_EQ(out[i], 0);
    free(gx); free(gy); free(out);
}

TEST(GradientMagnitude, ZeroGradZeroMagnitude_L2) {
    const int W = 32, H = 32;
    int16_t* gx  = alloc_grad(W, H);
    int16_t* gy  = alloc_grad(W, H);
    uint8_t* out = alloc_img(W, H, 0);
    gradient_magnitude_scalar(gx, gy, out, W, H, MagMethod::L2);
    for (int i = 0; i < W * H; i++)
        EXPECT_EQ(out[i], 0);
    free(gx); free(gy); free(out);
}

TEST(GradientMagnitude, OutputClamped) {
    const int W = 32, H = 32;
    int16_t* gx  = alloc_grad(W, H);
    int16_t* gy  = alloc_grad(W, H);
    uint8_t* out = alloc_img(W, H, 0);
    for (int i = 0; i < W * H; i++) { gx[i] = 1000; gy[i] = 1000; }
    gradient_magnitude_scalar(gx, gy, out, W, H, MagMethod::L1);
    for (int i = 0; i < W * H; i++) {
        EXPECT_GE(out[i], 0);
        EXPECT_LE(out[i], 255);
    }
    free(gx); free(gy); free(out);
}

TEST(GradientMagnitude, L1AndL2NonZeroOnRealGradient) {
    const int W = 32, H = 32;
    int16_t* gx    = alloc_grad(W, H);
    int16_t* gy    = alloc_grad(W, H);
    uint8_t* outL1 = alloc_img(W, H, 0);
    uint8_t* outL2 = alloc_img(W, H, 0);
    for (int i = 0; i < W * H; i++) gx[i] = 100;
    gradient_magnitude_scalar(gx, gy, outL1, W, H, MagMethod::L1);
    gradient_magnitude_scalar(gx, gy, outL2, W, H, MagMethod::L2);
    bool anyL1 = false, anyL2 = false;
    for (int i = 0; i < W * H; i++) {
        if (outL1[i] > 0) anyL1 = true;
        if (outL2[i] > 0) anyL2 = true;
    }
    EXPECT_TRUE(anyL1);
    EXPECT_TRUE(anyL2);
    free(gx); free(gy); free(outL1); free(outL2);
}

// =============================================================
// STAGE 2c - GRADIENT DIRECTION
// =============================================================

TEST(GradientDirection, ZeroGradDefaultsToZero) {
    const int W = 16, H = 16;
    int16_t* gx  = alloc_grad(W, H);
    int16_t* gy  = alloc_grad(W, H);
    uint8_t* dir = alloc_img(W, H, 0);
    gradient_direction_scalar(gx, gy, dir, W, H);
    for (int i = 0; i < W * H; i++)
        EXPECT_EQ(dir[i], 0);
    free(gx); free(gy); free(dir);
}

TEST(GradientDirection, PureHorizontalIsZeroDegrees) {
    const int W = 16, H = 16;
    int16_t* gx  = alloc_grad(W, H);
    int16_t* gy  = alloc_grad(W, H);
    uint8_t* dir = alloc_img(W, H, 0);
    for (int i = 0; i < W * H; i++) { gx[i] = 100; gy[i] = 0; }
    gradient_direction_scalar(gx, gy, dir, W, H);
    for (int i = 0; i < W * H; i++)
        EXPECT_EQ(dir[i], 0);
    free(gx); free(gy); free(dir);
}

TEST(GradientDirection, PureVerticalIs90Degrees) {
    const int W = 16, H = 16;
    int16_t* gx  = alloc_grad(W, H);
    int16_t* gy  = alloc_grad(W, H);
    uint8_t* dir = alloc_img(W, H, 0);
    for (int i = 0; i < W * H; i++) { gx[i] = 0; gy[i] = 100; }
    gradient_direction_scalar(gx, gy, dir, W, H);
    for (int i = 0; i < W * H; i++)
        EXPECT_EQ(dir[i], 90);
    free(gx); free(gy); free(dir);
}

TEST(GradientDirection, PositiveDiagonalIs45Degrees) {
    const int W = 16, H = 16;
    int16_t* gx  = alloc_grad(W, H);
    int16_t* gy  = alloc_grad(W, H);
    uint8_t* dir = alloc_img(W, H, 0);
    for (int i = 0; i < W * H; i++) { gx[i] = 100; gy[i] = 100; }
    gradient_direction_scalar(gx, gy, dir, W, H);
    for (int i = 0; i < W * H; i++)
        EXPECT_EQ(dir[i], 45);
    free(gx); free(gy); free(dir);
}

TEST(GradientDirection, NegativeDiagonalIs135Degrees) {
    const int W = 16, H = 16;
    int16_t* gx  = alloc_grad(W, H);
    int16_t* gy  = alloc_grad(W, H);
    uint8_t* dir = alloc_img(W, H, 0);
    for (int i = 0; i < W * H; i++) { gx[i] = 100; gy[i] = -100; }
    gradient_direction_scalar(gx, gy, dir, W, H);
    for (int i = 0; i < W * H; i++)
        EXPECT_EQ(dir[i], 135);
    free(gx); free(gy); free(dir);
}

TEST(GradientDirection, OnlyValidAnglesOutput) {
    const int W = 32, H = 32;
    int16_t* gx  = alloc_grad(W, H);
    int16_t* gy  = alloc_grad(W, H);
    uint8_t* dir = alloc_img(W, H, 0);
    for (int i = 0; i < W * H; i++) {
        gx[i] = (int16_t)(i % 300 - 150);
        gy[i] = (int16_t)((i * 3) % 300 - 150);
    }
    gradient_direction_scalar(gx, gy, dir, W, H);
    for (int i = 0; i < W * H; i++) {
        uint8_t d = dir[i];
        EXPECT_TRUE(d == 0 || d == 45 || d == 90 || d == 135)
            << "Invalid direction " << (int)d << " at pixel " << i;
    }
    free(gx); free(gy); free(dir);
}

// =============================================================
// STAGE 3 - NON-MAXIMUM SUPPRESSION
// =============================================================

TEST(NMS, ZeroMagnitudeStaysZero) {
    const int W = 32, H = 32;
    uint8_t* mag = alloc_img(W, H, 0);
    uint8_t* dir = alloc_img(W, H, 0);
    uint8_t* out = alloc_img(W, H, 99);
    non_max_suppression(mag, dir, out, W, H);
    for (int i = 0; i < W * H; i++)
        EXPECT_EQ(out[i], 0);
    free(mag); free(dir); free(out);
}

TEST(NMS, UniformMagnitudeKeepsAllInteriorPixels) {
    const int W = 32, H = 32;
    uint8_t* mag = alloc_img(W, H, 100);
    uint8_t* dir = alloc_img(W, H, 0);
    uint8_t* out = alloc_img(W, H, 0);
    non_max_suppression(mag, dir, out, W, H);
    for (int y = 1; y < H - 1; y++)
        for (int x = 1; x < W - 1; x++)
            EXPECT_EQ(out[y * W + x], 100);
    free(mag); free(dir); free(out);
}

TEST(NMS, LocalPeakSurvives) {
    const int W = 16, H = 16;
    uint8_t* mag = alloc_img(W, H, 50);
    uint8_t* dir = alloc_img(W, H, 0);
    uint8_t* out = alloc_img(W, H, 0);
    int cx = 8, cy = 8;
    mag[cy * W + cx] = 200;
    non_max_suppression(mag, dir, out, W, H);
    EXPECT_EQ(out[cy * W + cx], 200)   << "Peak should survive";
    EXPECT_EQ(out[cy * W + cx - 1], 0) << "Left neighbour should be suppressed";
    EXPECT_EQ(out[cy * W + cx + 1], 0) << "Right neighbour should be suppressed";
    free(mag); free(dir); free(out);
}

TEST(NMS, BorderIsAlwaysZero) {
    const int W = 16, H = 16;
    uint8_t* mag = alloc_img(W, H, 200);
    uint8_t* dir = alloc_img(W, H, 0);
    uint8_t* out = alloc_img(W, H, 0);
    non_max_suppression(mag, dir, out, W, H);
    for (int x = 0; x < W; x++) {
        EXPECT_EQ(out[x], 0);
        EXPECT_EQ(out[(H-1)*W+x], 0);
    }
    for (int y = 0; y < H; y++) {
        EXPECT_EQ(out[y*W], 0);
        EXPECT_EQ(out[y*W + W-1], 0);
    }
    free(mag); free(dir); free(out);
}

// =============================================================
// STAGE 4 - DOUBLE THRESHOLD
// =============================================================

TEST(DoubleThreshold, AboveHighIsStrong) {
    const int W = 16, H = 16;
    uint8_t* in  = alloc_img(W, H, 200);
    uint8_t* out = alloc_img(W, H, 0);
    double_threshold(in, out, W, H, 50, 100);
    for (int i = 0; i < W * H; i++)
        EXPECT_EQ(out[i], 255);
    free(in); free(out);
}

TEST(DoubleThreshold, BelowLowIsSuppressed) {
    const int W = 16, H = 16;
    uint8_t* in  = alloc_img(W, H, 10);
    uint8_t* out = alloc_img(W, H, 99);
    double_threshold(in, out, W, H, 50, 100);
    for (int i = 0; i < W * H; i++)
        EXPECT_EQ(out[i], 0);
    free(in); free(out);
}

TEST(DoubleThreshold, BetweenThresholdsIsWeak) {
    const int W = 16, H = 16;
    uint8_t* in  = alloc_img(W, H, 75);
    uint8_t* out = alloc_img(W, H, 0);
    double_threshold(in, out, W, H, 50, 100);
    for (int i = 0; i < W * H; i++)
        EXPECT_EQ(out[i], 128);
    free(in); free(out);
}

TEST(DoubleThreshold, OnlyValidOutputValues) {
    const int W = 32, H = 32;
    uint8_t* in  = alloc_img(W, H, 0);
    uint8_t* out = alloc_img(W, H, 0);
    for (int i = 0; i < W * H; i++) in[i] = (uint8_t)(i % 256);
    double_threshold(in, out, W, H, 64, 192);
    for (int i = 0; i < W * H; i++) {
        uint8_t v = out[i];
        EXPECT_TRUE(v == 0 || v == 128 || v == 255)
            << "Invalid value " << (int)v << " at pixel " << i;
    }
    free(in); free(out);
}

// =============================================================
// STAGE 5 - HYSTERESIS TRACING
// =============================================================

TEST(Hysteresis, OnlyStrongPixelsUnchanged) {
    const int W = 16, H = 16;
    uint8_t* img = alloc_img(W, H, 255);
    hysteresis_tracing(img, W, H);
    for (int i = 0; i < W * H; i++)
        EXPECT_EQ(img[i], 255);
    free(img);
}

TEST(Hysteresis, WeakPixelsWithNoStrongAreSuppressed) {
    const int W = 16, H = 16;
    uint8_t* img = alloc_img(W, H, 128);
    hysteresis_tracing(img, W, H);
    for (int i = 0; i < W * H; i++)
        EXPECT_EQ(img[i], 0);
    free(img);
}

TEST(Hysteresis, WeakPixelAdjacentToStrongIsPromoted) {
    const int W = 16, H = 16;
    uint8_t* img = alloc_img(W, H, 0);
    img[8 * W + 8] = 255;
    img[8 * W + 9] = 128;
    hysteresis_tracing(img, W, H);
    EXPECT_EQ(img[8 * W + 8], 255) << "Strong pixel should stay strong";
    EXPECT_EQ(img[8 * W + 9], 255) << "Adjacent weak should be promoted";
    free(img);
}

TEST(Hysteresis, IsolatedWeakPixelIsSuppressed) {
    const int W = 16, H = 16;
    uint8_t* img = alloc_img(W, H, 0);
    img[4 * W + 4] = 128;
    hysteresis_tracing(img, W, H);
    EXPECT_EQ(img[4 * W + 4], 0) << "Isolated weak pixel should be suppressed";
    free(img);
}

TEST(Hysteresis, ChainOfWeakPixelsConnectedToStrongArePromoted) {
    const int W = 32, H = 32;
    uint8_t* img = alloc_img(W, H, 0);
    img[5 * W + 5] = 255;
    img[5 * W + 6] = 128;
    img[5 * W + 7] = 128;
    img[5 * W + 8] = 128;
    img[5 * W + 9] = 128;
    hysteresis_tracing(img, W, H);
    EXPECT_EQ(img[5 * W + 5], 255) << "Anchor should stay strong";
    EXPECT_EQ(img[5 * W + 6], 255) << "Chain pixel 1 promoted";
    EXPECT_EQ(img[5 * W + 7], 255) << "Chain pixel 2 promoted";
    EXPECT_EQ(img[5 * W + 8], 255) << "Chain pixel 3 promoted";
    EXPECT_EQ(img[5 * W + 9], 255) << "Chain pixel 4 promoted";
    free(img);
}

// =============================================================
// FULL PIPELINE INTEGRATION TESTS
// =============================================================

TEST(FullPipeline, UniformImageProducesNoEdges) {
    const int W = 64, H = 64;
    uint8_t* in      = alloc_img(W, H, 150);
    uint8_t* blurred = alloc_img(W, H, 0);
    int16_t* gx      = alloc_grad(W, H);
    int16_t* gy      = alloc_grad(W, H);
    uint8_t* mag     = alloc_img(W, H, 0);
    uint8_t* dir     = alloc_img(W, H, 0);
    uint8_t* nms_out = alloc_img(W, H, 0);
    uint8_t* thresh  = alloc_img(W, H, 0);

    gaussian_blur_scalar(in, blurred, W, H);
    sobel_gradients_scalar(blurred, gx, gy, W, H);
    gradient_magnitude_scalar(gx, gy, mag, W, H, MagMethod::L1);
    gradient_direction_scalar(gx, gy, dir, W, H);
    non_max_suppression(mag, dir, nms_out, W, H);
    double_threshold(nms_out, thresh, W, H, 10, 30);
    hysteresis_tracing(thresh, W, H);

    // Zero-padding creates an artificial contrast at the outer image border.
    // The uniform interior should still produce no edges.
    for (int y = 4; y < H - 4; y++)
        for (int x = 4; x < W - 4; x++)
            EXPECT_EQ(thresh[y * W + x], 0) << "Unexpected edge at pixel " << (y * W + x);

    free(in); free(blurred); free(gx); free(gy);
    free(mag); free(dir); free(nms_out); free(thresh);
}

TEST(FullPipeline, VerticalEdgeDetectedEndToEnd) {
    const int W = 64, H = 64;
    uint8_t* in      = alloc_img(W, H, 0);
    uint8_t* blurred = alloc_img(W, H, 0);
    int16_t* gx      = alloc_grad(W, H);
    int16_t* gy      = alloc_grad(W, H);
    uint8_t* mag     = alloc_img(W, H, 0);
    uint8_t* dir     = alloc_img(W, H, 0);
    uint8_t* nms_out = alloc_img(W, H, 0);
    uint8_t* thresh  = alloc_img(W, H, 0);

    for (int y = 0; y < H; y++)
        for (int x = W / 2; x < W; x++)
            in[y * W + x] = 255;

    gaussian_blur_scalar(in, blurred, W, H);
    sobel_gradients_scalar(blurred, gx, gy, W, H);
    gradient_magnitude_scalar(gx, gy, mag, W, H, MagMethod::L2);
    gradient_direction_scalar(gx, gy, dir, W, H);
    non_max_suppression(mag, dir, nms_out, W, H);
    double_threshold(nms_out, thresh, W, H, 10, 30);
    hysteresis_tracing(thresh, W, H);

    bool edge_found = false;
    for (int y = 2; y < H - 2; y++)
        if (thresh[y * W + W / 2] > 0) edge_found = true;

    EXPECT_TRUE(edge_found) << "No edge detected after full 5-stage pipeline";

    free(in); free(blurred); free(gx); free(gy);
    free(mag); free(dir); free(nms_out); free(thresh);
}


TEST(FullPipeline, NonPowerOfTwoImageSize) {
    const int W = 100, H = 75;
    uint8_t* in      = alloc_img(W, H, 0);
    uint8_t* blurred = alloc_img(W, H, 0);
    int16_t* gx      = alloc_grad(W, H);
    int16_t* gy      = alloc_grad(W, H);
    uint8_t* mag     = alloc_img(W, H, 0);
    uint8_t* dir     = alloc_img(W, H, 0);
    uint8_t* nms_out = alloc_img(W, H, 0);
    uint8_t* thresh  = alloc_img(W, H, 0);

    for (int y = 0; y < H; y++)
        for (int x = W / 2; x < W; x++)
            in[y * W + x] = 255;

    gaussian_blur_scalar(in, blurred, W, H);
    sobel_gradients_scalar(blurred, gx, gy, W, H);
    gradient_magnitude_scalar(gx, gy, mag, W, H, MagMethod::L2);
    gradient_direction_scalar(gx, gy, dir, W, H);
    non_max_suppression(mag, dir, nms_out, W, H);
    double_threshold(nms_out, thresh, W, H, 10, 30);
    hysteresis_tracing(thresh, W, H);

    bool edge_found = false;
    for (int y = 2; y < H - 2; y++)
        if (thresh[y * W + W / 2] > 0) edge_found = true;

    EXPECT_TRUE(edge_found) << "Non-power-of-two image should still detect the edge";

    free(in); free(blurred); free(gx); free(gy);
    free(mag); free(dir); free(nms_out); free(thresh);
}
// =============================================================
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
