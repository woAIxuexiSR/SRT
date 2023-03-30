#pragma once

#include <cuda_runtime.h>

#include "helper_cuda.h"
#include "helper_math.h"
#include "definition.h"

class Film
{
private:
    int width, height;
    GPUMemory<float4> pixels_f;
    GPUMemory<uchar4> pixels_u;

public:

    Film(int _w, int _h): width(_w), height(_h), pixels_f(_w* _h), pixels_u(_w* _h) {}

    float4* get_fptr() { return pixels_f.data(); }

    uchar4* get_uptr() { return pixels_u.data(); }

    int get_width() const { return width; }

    int get_height() const { return height; }

    void resize(int w, int h);

    void memset_f0();

    void f_to_uchar();

    void save_png(const string& filename) const;

    void save_jpg(const string& filename) const;

    void save_exr(const string& filename) const;
};