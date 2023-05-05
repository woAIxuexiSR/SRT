#pragma once

#include "definition.h"
#include "my_math.h"

class Texture
{
public:
    vector<uchar4> pixels;
    uint2 resolution;

public:
    Texture() : resolution({ 0, 0 }) {}
    
    void load_from_file(const string& filename);
};