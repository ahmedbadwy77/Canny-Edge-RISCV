#ifndef HYSTERESIS_H
#define HYSTERESIS_H

#include <cstdint>
#include <stack>
#include <utility>

void hysteresis_tracing(uint8_t* image, int width, int height) {
    const uint8_t STRONG = 255;
    const uint8_t WEAK   = 128;

    // 8-connected neighbour offsets
    const int dx[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
    const int dy[8] = {-1, -1, -1, 0, 0, 1, 1, 1};

    std::stack<std::pair<int,int>> stk;

    // Seed the stack with all strong pixels
    for (int y = 1; y < height - 1; y++)
        for (int x = 1; x < width - 1; x++)
            if (image[y * width + x] == STRONG)
                stk.push({x, y});

    // Flood fill outward from strong pixels
    while (!stk.empty()) {
        std::pair<int, int> current = stk.top();
        stk.pop();
        int cx = current.first;
        int cy = current.second;

        for (int d = 0; d < 8; d++) {
            int nx = cx + dx[d];
            int ny = cy + dy[d];

            if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;

            if (image[ny * width + nx] == WEAK) {
                // Promote this weak pixel to strong
                image[ny * width + nx] = STRONG;
                stk.push({nx, ny});
            }
        }
    }

    // Suppress all remaining weak pixels that were never reached
    for (int i = 0; i < width * height; i++)
        if (image[i] == WEAK) image[i] = 0;
}

#endif
