#include "../include/image_io.h"
#include <cstdio>
#include <cstdlib>


uint8_t* read_raw_image(const char* filename, int width, int height) {
    if (width <= 0 || height <= 0) return nullptr;

    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error: Could not open file %s\n", filename);
        return nullptr;
    }

    size_t size = static_cast<size_t>(width) * static_cast<size_t>(height);
    uint8_t* data = static_cast<uint8_t*>(malloc(size));
    if (!data) {
        fclose(file);
        return nullptr;
    }

    size_t bytes_read = fread(data, 1, size, file);
    fclose(file);
    if (bytes_read != size) {
        printf("Error: Expected %zu bytes from %s, read %zu\n", size, filename, bytes_read);
        free(data);
        return nullptr;
    }

    return data;
}


bool write_raw_image(const char* filename, const uint8_t* data, int width, int height) {
    if (!data || width <= 0 || height <= 0) return false;

    FILE* file = fopen(filename, "wb");
    if (!file) {
        printf("Error: Could not create file %s\n", filename);
        return false;
    }

    size_t size = static_cast<size_t>(width) * static_cast<size_t>(height);
    size_t bytes_written = fwrite(data, 1, size, file);
    bool closed = fclose(file) == 0;
    if (bytes_written != size || !closed) {
        printf("Error: Could not write complete file %s\n", filename);
        return false;
    }
    return true;
}
