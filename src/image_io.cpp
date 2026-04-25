#include "../include/image_io.h"
#include <cstdio>
#include <cstdlib>


uint8_t* read_raw_image(const char* filename, int width, int height) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error: Could not open file %s\n", filename);
        return nullptr;
    }

    size_t size = width * height;

    uint8_t* data = (uint8_t*)aligned_alloc(32, size);
    
    if (data) {
        fread(data, 1, size, file);
    }
    
    fclose(file);
    return data;
}


void write_raw_image(const char* filename, uint8_t* data, int width, int height) {
    if (!data) return;

    FILE* file = fopen(filename, "wb");
    if (!file) {
        printf("Error: Could not create file %s\n", filename);
        return;
    }

    fwrite(data, 1, width * height, file);
    fclose(file);
}