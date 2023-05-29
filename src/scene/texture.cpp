#include "texture.h"

void Texture::load_from_file(const string& filename)
{
    void* data = nullptr;
    if (IsEXR(filename.c_str()) == TINYEXR_SUCCESS)
    {
        int res = LoadEXR((float**)&data, (int*)&resolution.x, (int*)&resolution.y, filename.c_str(), nullptr);
        if (res != TINYEXR_SUCCESS)
        {
            cout << "ERROR::Failed to Load EXR file: " << filename << endl;
            exit(-1);
        }
        format = Format::Float;
    }
    else if (stbi_is_hdr(filename.c_str()))
    {
        data = stbi_loadf(filename.c_str(), (int*)&resolution.x, (int*)&resolution.y, nullptr, STBI_rgb_alpha);
        if (!data)
        {
            cout << "ERROR::Failed to Load HDR file: " << filename << endl;
            exit(-1);
        }
        format = Format::Float;
    }
    else
    {
        data = stbi_load(filename.c_str(), (int*)&resolution.x, (int*)&resolution.y, nullptr, STBI_rgb_alpha);
        if (!data)
        {
            cout << "ERROR::Failed to Load LDR file: " << filename << endl;
            exit(-1);
        }
        format = Format::Uchar;
    }

    if (format == Format::Float)
    {
        pixels_f.resize(resolution.x * resolution.y);
        memcpy(pixels_f.data(), data, resolution.x * resolution.y * sizeof(float4));
    }
    else
    {
        pixels_u.resize(resolution.x * resolution.y);
        memcpy(pixels_u.data(), data, resolution.x * resolution.y * sizeof(uchar4));
    }
    free(data);
}

void* Texture::get_pixels()
{
    if (format == Format::Float)
        return pixels_f.data();
    return pixels_u.data();
}