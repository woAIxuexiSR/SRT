#pragma once

#include "definition.h"
#include "my_math.h"
#include "helper_cuda.h"

class Film
{
private:
    int width, height;
    GPUMemory<float4> pixels;

public:
    Film(int _w, int _h)
        : width(_w), height(_h), pixels(_w* _h) {}

    int get_width() const { return width; }
    int get_height() const { return height; }
    float4* get_pixels() { return pixels.data(); }
};