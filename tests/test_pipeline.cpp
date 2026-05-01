#include <gtest/gtest.h>
#include <cstdint>
#include <cmath>
#include <cstdlib>

#include "../include/gaussian.h"
#include "../include/sobel.h"

// ─────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────

static uint8_t* alloc_image(int w, int h, uint8_t fill = 0) {
    uint8_t* img = (uint8_t*)aligned_alloc(32, w * h);
    for (int i = 0; i < w * h; i++) img[i] = fill;
    return img;
}

static int16_t* alloc_grad(int w, int h) {
    return (int16_t*)aligned_alloc(32, w * h * sizeof(int16_t));
}

// ─────────────────────────────────────────────
// GAUSSIAN BLUR TESTS
// ─────────────────────────────────────────────

// A uniform image blurred must stay the same constant value
TEST(GaussianBlur, UniformImageStaysConstant) {
    const int W = 64, H = 64;
    uint8_t* input  = alloc_image(W, H, 128);
    uint8_t* output = alloc_image(W, H, 0);

    gaussian_blur_scalar(input, output, W, H);

    for (int i = 0; i < W * H; i++)
        EXPECT_EQ(output[i], 128) << "Failed at pixel " << i;

    free(input); free(output);
}

// Zero image in → zero image out
TEST(GaussianBlur, ZeroImageStaysZero) {
    const int W = 32, H = 32;
    uint8_t* input  = alloc_image(W, H, 0);
    uint8_t* output = alloc_image(W, H, 0);

    gaussian_blur_scalar(input, output, W, H);

    for (int i = 0; i < W * H; i++)
        EXPECT_EQ(output[i], 0) << "Failed at pixel " << i;

    free(input); free(output);
}

// Output must stay within [0, 255]
TEST(GaussianBlur, OutputClamped) {
    const int W = 64, H = 64;
    uint8_t* input  = alloc_image(W, H, 255);
    uint8_t* output = alloc_image(W, H, 0);

    gaussian_blur_scalar(input, output, W, H);

    for (int i = 0; i < W * H; i++) {
        EXPECT_GE(output[i], 0);
        EXPECT_LE(output[i], 255);
    }

    free(input); free(output);
}

// A single bright pixel (impulse) must produce a blurred response
// that is strictly less bright than the original
TEST(GaussianBlur, ImpulseResponseIsSmoothed) {
    const int W = 32, H = 32;
    uint8_t* input  = alloc_image(W, H, 0);
    uint8_t* output = alloc_image(W, H, 0);

    // Place impulse at center
    input[16 * W + 16] = 255;

    gaussian_blur_scalar(input, output, W, H);

    // Center pixel should be less than 255 (spread out)
    EXPECT_LT(output[16 * W + 16], 255);
    // And greater than 0 (energy preserved near center)
    EXPECT_GT(output[16 * W + 16], 0);

    free(input); free(output);
}

// ─────────────────────────────────────────────
// SOBEL GRADIENT TESTS
// ─────────────────────────────────────────────

// Uniform image → zero gradient everywhere
TEST(SobelGradients, UniformImageZeroGradient) {
    const int W = 64, H = 64;
    uint8_t* input  = alloc_image(W, H, 100);
    int16_t* gx     = alloc_grad(W, H);
    int16_t* gy     = alloc_grad(W, H);

    sobel_gradients_scalar(input, gx, gy, W, H);

    // Interior pixels (away from border) must be exactly zero
    for (int y = 1; y < H - 1; y++)
        for (int x = 1; x < W - 1; x++) {
            EXPECT_EQ(gx[y * W + x], 0) << "Gx non-zero at (" << x << "," << y << ")";
            EXPECT_EQ(gy[y * W + x], 0) << "Gy non-zero at (" << x << "," << y << ")";
        }

    free(input); free(gx); free(gy);
}

// Vertical edge: left half black, right half white → strong Gx, weak Gy
TEST(SobelGradients, VerticalEdgeStrongGx) {
    const int W = 64, H = 64;
    uint8_t* input  = alloc_image(W, H, 0);
    int16_t* gx     = alloc_grad(W, H);
    int16_t* gy     = alloc_grad(W, H);

    // Right half = 255
    for (int y = 0; y < H; y++)
        for (int x = W / 2; x < W; x++)
            input[y * W + x] = 255;

    sobel_gradients_scalar(input, gx, gy, W, H);

    // At the edge column (x = W/2), interior rows should have strong Gx
    int edge_x = W / 2;
    for (int y = 1; y < H - 1; y++) {
        EXPECT_GT(std::abs(gx[y * W + edge_x]), 0)
            << "Expected non-zero Gx at vertical edge, row " << y;
    }

    free(input); free(gx); free(gy);
}

// Horizontal edge: top half black, bottom half white → strong Gy, weak Gx
TEST(SobelGradients, HorizontalEdgeStrongGy) {
    const int W = 64, H = 64;
    uint8_t* input  = alloc_image(W, H, 0);
    int16_t* gx     = alloc_grad(W, H);
    int16_t* gy     = alloc_grad(W, H);

    // Bottom half = 255
    for (int y = H / 2; y < H; y++)
        for (int x = 0; x < W; x++)
            input[y * W + x] = 255;

    sobel_gradients_scalar(input, gx, gy, W, H);

    int edge_y = H / 2;
    for (int x = 1; x < W - 1; x++) {
        EXPECT_GT(std::abs(gy[edge_y * W + x]), 0)
            << "Expected non-zero Gy at horizontal edge, col " << x;
    }

    free(input); free(gx); free(gy);
}

// Zero image → zero gradients
TEST(SobelGradients, ZeroImageZeroGradient) {
    const int W = 32, H = 32;
    uint8_t* input  = alloc_image(W, H, 0);
    int16_t* gx     = alloc_grad(W, H);
    int16_t* gy     = alloc_grad(W, H);

    sobel_gradients_scalar(input, gx, gy, W, H);

    for (int i = 0; i < W * H; i++) {
        EXPECT_EQ(gx[i], 0);
        EXPECT_EQ(gy[i], 0);
    }

    free(input); free(gx); free(gy);
}

// ─────────────────────────────────────────────
// PIPELINE INTEGRATION TEST
// ─────────────────────────────────────────────

// Blur then Sobel on uniform image → zero gradients
TEST(Pipeline, BlurThenSobelUniformIsZero) {
    const int W = 64, H = 64;
    uint8_t* input   = alloc_image(W, H, 200);
    uint8_t* blurred = alloc_image(W, H, 0);
    int16_t* gx      = alloc_grad(W, H);
    int16_t* gy      = alloc_grad(W, H);

    gaussian_blur_scalar(input, blurred, W, H);
    sobel_gradients_scalar(blurred, gx, gy, W, H);

    for (int y = 1; y < H - 1; y++)
        for (int x = 1; x < W - 1; x++) {
            EXPECT_EQ(gx[y * W + x], 0);
            EXPECT_EQ(gy[y * W + x], 0);
        }

    free(input); free(blurred); free(gx); free(gy);
}

// ─────────────────────────────────────────────
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
