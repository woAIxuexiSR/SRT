#pragma once

#include <cuda_runtime.h>
#include <string>
#include <vector>
#include <iostream>

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

    Film(int _w, int _h) : width(_w), height(_h), pixels_f(_w * _h), pixels_u(_w * _h) {}

    float4* getfPtr() { return pixels_f.data(); }

    uchar4* getuPtr() { return pixels_u.data(); }

    int getWidth() const { return width; }
    
    int getHeight() const { return height; }

    void fToUchar();

    void save_png(const std::string& filename) const;

    void save_jpg(const std::string& filename) const;

    void save_exr(const std::string& filename) const;

    void memset_0();
};