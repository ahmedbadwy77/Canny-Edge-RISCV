#ifndef IMAGE_IO_H
#define IMAGE_IO_H

#include <cstdint>

uint8_t* read_raw_image(const char* filename, int width, int height);


void write_raw_image(const char* filename, uint8_t* data, int width, int height);

#endif