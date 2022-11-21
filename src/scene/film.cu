#include "film.h"

__global__ void fToUchar_k(int n_elements, float4* __restrict__ src, uchar4* __restrict__ dst)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_elements) return;
    float4 clamped = clamp(src[i], 0.0f, 1.0f);
    float w = 1.0f / 2.2f;
    float4 hdr = make_float4(powf(clamped.x, w), powf(clamped.y, w), powf(clamped.z, w), powf(clamped.w, w));
    float4 v = hdr * 255.f;
    dst[i] = make_uchar4(v.x, v.y, v.z, v.w);
}

__global__ void flipfVertical_k(int n_elements, int width, int height, float4* __restrict__ src, float4* __restrict__ dst)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_elements) return;
    int x = i % width, y = i / width;
    dst[i] = src[(height - y - 1) * width + x];
}

void Film::fToUchar()
{
    int pixel_num = width * height;
    tcnn::linear_kernel(fToUchar_k, 0, 0, pixel_num, pixels_f.data(), pixels_u.data());
    checkCudaErrors(cudaDeviceSynchronize());
}

void Film::save_jpg(const std::string& filename) const
{
    std::vector<uchar4> pixels_cpu(width * height);
    pixels_u.copy_to_host(pixels_cpu);

    stbi_flip_vertically_on_write(true);
    int ret = stbi_write_jpg(filename.c_str(), width, height, 4, (void*)pixels_cpu.data(), 100);
    if (ret == 0)
        std::cout << "Failed to save image: " << filename << std::endl;
}

void Film::save_exr(const std::string& filename) const
{
    EXRHeader header;
    InitEXRHeader(&header);
    EXRImage image;
    InitEXRImage(&image);

    GPUMemory<float4> flipped(width * height);
    tcnn::linear_kernel(flipfVertical_k, 0, 0, width * height, width, height, pixels_f.data(), flipped.data());
    std::vector<float4> pixels_cpu(width * height);
    flipped.copy_to_host(pixels_cpu);

    std::vector<float> images[4]{
        std::vector<float>(width * height),
        std::vector<float>(width * height),
        std::vector<float>(width * height),
        std::vector<float>(width * height)
    };
    for (int i = 0; i < width * height; i++)
    {
        images[0][i] = pixels_cpu[i].x;
        images[1][i] = pixels_cpu[i].y;
        images[2][i] = pixels_cpu[i].z;
        images[3][i] = pixels_cpu[i].w;
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
        std::cout << "Failed to save image: " << filename << std::endl;

    free(header.channels);
    free(header.pixel_types);
    free(header.requested_pixel_types);
}