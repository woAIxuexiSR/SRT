#include "texture.h"

void Texture::load_from_file(const string& filename)
{
    uint2 res;
    int compontents;
    unsigned char* data = stbi_load(filename.c_str(), (int*)&res.x, (int*)&res.y, &compontents, STBI_rgb_alpha);

    if (!data)
    {
        cout << "ERROR::TEXTURE::LOAD_FROM_FILE::" << filename << endl;
        return;
    }

    pixels.resize(res.x * res.y);
    memcpy(pixels.data(), data, res.x * res.y * sizeof(uchar4));
    resolution = res;
}