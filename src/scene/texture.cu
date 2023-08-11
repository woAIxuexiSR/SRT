#include "texture.h"

Image::Image(int _w, int _h, float4* data)
{
    resolution = make_uint2(_w, _h);
    pixels_f.resize(resolution.x * resolution.y);
    checkCudaErrors(cudaMemcpy(pixels_f.data(), data, sizeof(float4) * pixels_f.size(), cudaMemcpyDeviceToHost));
    format = Format::Float;
}

void Image::f_to_u()
{
    if (format == Format::Uchar) return;

    pixels_u.resize(resolution.x * resolution.y);
    for (int i = 0; i < pixels_u.size(); i++)
    {
        float4 f = clamp(pixels_f[i], 0.0f, 1.0f) * 255.0f;
        pixels_u[i] = make_uchar4(f.x, f.y, f.z, f.w);
    }
}

void Image::u_to_f()
{
    if (format == Format::Float) return;

    pixels_f.resize(resolution.x * resolution.y);
    for (int i = 0; i < pixels_f.size(); i++)
    {
        uchar4 u = pixels_u[i];
        pixels_f[i] = make_float4(u.x, u.y, u.z, u.w) / 255.0f;
    }
}

void Image::flip(bool x, bool y)
{
    if (x)
    {
        for (int i = 0; i < resolution.x / 2; i++)
        {
            for (int j = 0; j < resolution.y; j++)
            {
                int idx1 = i + j * resolution.x;
                int idx2 = (resolution.x - i - 1) + j * resolution.x;
                if (format == Format::Float)
                    std::swap(pixels_f[idx1], pixels_f[idx2]);
                else
                    std::swap(pixels_u[idx1], pixels_u[idx2]);
            }
        }
    }
    if (y)
    {
        for (int i = 0; i < resolution.x; i++)
        {
            for (int j = 0; j < resolution.y / 2; j++)
            {
                int idx1 = i + j * resolution.x;
                int idx2 = i + (resolution.y - j - 1) * resolution.x;
                if (format == Format::Float)
                    std::swap(pixels_f[idx1], pixels_f[idx2]);
                else
                    std::swap(pixels_u[idx1], pixels_u[idx2]);
            }
        }
    }
}

void Image::save_exr(const string& filename)
{
    u_to_f();

    int width = resolution.x;
    int height = resolution.y;

    EXRHeader header;
    InitEXRHeader(&header);
    EXRImage image;
    InitEXRImage(&image);

    vector<float> images[4]{
        vector<float>(width * height),
        vector<float>(width * height),
        vector<float>(width * height),
        vector<float>(width * height)
    };
    for (int i = 0; i < width * height; i++)
    {
        images[0][i] = pixels_f[i].x;
        images[1][i] = pixels_f[i].y;
        images[2][i] = pixels_f[i].z;
        images[3][i] = pixels_f[i].w;
    }

    float* image_ptr[4];
    image_ptr[0] = &(images[2][0]);
    image_ptr[1] = &(images[1][0]);
    image_ptr[2] = &(images[0][0]);
    image_ptr[3] = &(images[3][0]);

    image.num_channels = 4;
    image.images = (unsigned char**)image_ptr;
    image.width = width;
    image.height = height;
    header.num_channels = 4;
    header.channels = (EXRChannelInfo*)malloc(sizeof(EXRChannelInfo) * header.num_channels);
    strncpy(header.channels[0].name, "B", 255); header.channels[0].name[strlen("B")] = '\0';
    strncpy(header.channels[1].name, "G", 255); header.channels[1].name[strlen("G")] = '\0';
    strncpy(header.channels[2].name, "R", 255); header.channels[2].name[strlen("R")] = '\0';
    strncpy(header.channels[3].name, "A", 255); header.channels[3].name[strlen("A")] = '\0';
    header.pixel_types = (int*)malloc(sizeof(int) * header.num_channels);
    header.requested_pixel_types = (int*)malloc(sizeof(int) * header.num_channels);
    for (int i = 0; i < header.num_channels; i++)
    {
        header.pixel_types[i] = TINYEXR_PIXELTYPE_FLOAT;
        header.requested_pixel_types[i] = TINYEXR_PIXELTYPE_HALF;
    }

    const char* err;
    int ret = SaveEXRImageToFile(&image, &header, filename.c_str(), &err);
    if (ret != TINYEXR_SUCCESS)
    {
        cout << "ERROR::Failed to save image: " << filename << endl;
        exit(-1);
    }

    free(header.channels);
    free(header.pixel_types);
    free(header.requested_pixel_types);
}

void Image::save_hdr(const string& filename)
{
    u_to_f();

    int ret = stbi_write_hdr(filename.c_str(), resolution.x, resolution.y, 4, (float*)pixels_f.data());

    if (!ret)
    {
        cout << "ERROR::Failed to save image: " << filename << endl;
        exit(-1);
    }
}

void Image::save_ldr(const string& filename)
{
    f_to_u();

    string ext = filename.substr(filename.find_last_of(".") + 1);
    int ret = 0;
    if (ext == "png")
        ret = stbi_write_png(filename.c_str(), resolution.x, resolution.y, 4, (void*)pixels_u.data(), 0);
    else if (ext == "bmp")
        ret = stbi_write_bmp(filename.c_str(), resolution.x, resolution.y, 4, (void*)pixels_u.data());
    else if (ext == "jpg")
        ret = stbi_write_jpg(filename.c_str(), resolution.x, resolution.y, 4, (void*)pixels_u.data(), 100);
    else
    {
        cout << "ERROR::Unsupported image format: " << ext << endl;
        exit(-1);
    }

    if (!ret)
    {
        cout << "ERROR::Failed to save image: " << filename << endl;
        exit(-1);
    }
}

void Image::load_from_file(const string& filename, bool flip_y)
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
        if(flip_y) flip(false, true);
    }
    else
    {
        pixels_u.resize(resolution.x * resolution.y);
        memcpy(pixels_u.data(), data, resolution.x * resolution.y * sizeof(uchar4));
        if (flip_y) flip(false, true);
    }
    free(data);
}

void Image::save_to_file(const string& filename)
{
    flip(true, true);

    string ext = filename.substr(filename.find_last_of(".") + 1);
    if(ext == "exr")
        save_exr(filename);
    else if (ext == "hdr")
        save_hdr(filename);
    else
        save_ldr(filename);
}

void* Image::get_pixels()
{
    return (format == Format::Float) ? (void*)pixels_f.data() : (void*)pixels_u.data();
}