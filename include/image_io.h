#ifndef IMAGE_IO_H
#define IMAGE_IO_H

#include <cstdint>

uint8_t* read_raw_image(const char* filename, int width, int height);

bool write_raw_image(const char* filename, const uint8_t* data, int width, int height);

#endif
