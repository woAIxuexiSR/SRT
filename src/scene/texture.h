#pragma once

#include "definition.h"
#include "my_math.h"

class Image
{
public:
    enum class Format { Uchar, Float };

    Format format;
    uint2 resolution;
    vector<float4> pixels_f;
    vector<uchar4> pixels_u;

public:
    Image() : format(Format::Uchar), resolution({ 0, 0 }) {}

    /* helper functions */

    void f_to_u();
    void u_to_f();
    void save_exr(const string& filename);
    void save_hdr(const string& filename);
    void save_ldr(const string& filename);

    /* useful functions */

    void load_from_file(const string& filename);
    void save_to_file(const string& filename);
    void* get_pixels();
};


class Texture
{
public:
    string name;
    shared_ptr<Image> image;
};