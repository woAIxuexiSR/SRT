#pragma once

#include "definition.h"
#include "my_math.h"

class Texture
{
public:

    enum class Format
    {
        Float,
        Uchar
    };

    Format format;
    vector<float4> pixels_f;
    vector<uchar4> pixels_u;
    uint2 resolution;

public:
    Texture() : resolution({ 0, 0 }) {}
    
    void load_from_file(const string& filename);
    void* get_pixels();
};