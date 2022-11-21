#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
#include <chrono>

#include <cuda_runtime.h>
#include <cuda.h>
// #include <curand_kernel.h>
#include <curand.h>
#include "helper_cuda.h"

#include <optix.h>
#include <optix_stubs.h>

#include <tiny-cuda-nn/common_device.h>
#include <tiny-cuda-nn/config.h>

#include "defs.h"
#include "matrix.h"
#include "transform.h"
#include "bxdf.h"
#include "film.h"
#include "camera.h"
#include "scene.h"
#include "light.h"

#include "helper_optix.h"
#include "OptixApp.h"

#include "gui.h"

void STB_test()
{
    const int width = 800, height = 600, nchannels = 3;
    std::vector<uchar3> pixels(width * height);
    for (int i = 0; i < width * height; ++i)
    {
        pixels[i].x = ((i % width) / (float)width) * 255.0f;
        pixels[i].y = ((i / width) / (float)height) * 255.0f;
        pixels[i].z = 0;
    }

    const std::string filename("STB_test.jpg");
    int ret = stbi_write_jpg(filename.c_str(), width, height, nchannels, (void *)pixels.data(), 1);
    if (ret == 0)
        std::cout << "Error writing file " << filename << std::endl;

    int rw, rh, nc;
    unsigned char *data = stbi_load(filename.c_str(), &rw, &rh, &nc, 0);
    if (data == nullptr)
        std::cout << "Error loading file " << filename << std::endl;

    if (rw != width || rh != height || nc != nchannels)
        std::cout << "Error: file " << filename << " has wrong dimensions" << std::endl;

    stbi_image_free(data);
    std::cout << "Successfully tested STB" << std::endl;
}

void TinyExr_test()
{
    const int width = 800, height = 600;
    std::vector<float3> pixels(width * height);
    for (int i = 0; i < width * height; ++i)
    {
        pixels[i].x = (i % width) / (float)width;
        pixels[i].y = (i / width) / (float)height;
        pixels[i].z = 0;
    }

    const std::string filename("TinyExr_test.exr");
    bool ret = SaveEXR((float *)pixels.data(), width, height, filename.c_str());
    if (!ret)
        std::cout << "Error writing file " << filename << std::endl;

    std::cout << "Successfully tested TinyExr" << std::endl;
}

void OptiX_test()
{
    checkCudaErrors(cudaFree(0));
    int num_devices;
    checkCudaErrors(cudaGetDeviceCount(&num_devices));
    std::cout << "  Number of devices: " << num_devices << std::endl;

    OptixResult res = optixInit();
    if (res != OPTIX_SUCCESS)
        std::cout << "Error initializing OptiX" << std::endl;

    std::cout << "Successfully tested OptiX" << std::endl;
}

template <int stride>
__global__ void eval_image(uint32_t n_elements, cudaTextureObject_t texture, float *__restrict__ xs_and_ys, float *__restrict__ result)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_elements)
        return;

    uint32_t output_idx = i * stride;
    uint32_t input_idx = i * 2;

    float4 val = tex2D<float4>(texture, xs_and_ys[input_idx], xs_and_ys[input_idx + 1]);
    result[output_idx + 0] = val.x;
    result[output_idx + 1] = val.y;
    result[output_idx + 2] = val.z;
}

void TinyCudaNN_test()
{
    tcnn::json config = {
        {"loss", {{"otype", "RelativeL2"}}},
        {"optimizer", {
                          {"otype", "Adam"},
                          {"learning_rate", 1e-2},
                          {"beta1", 0.9f},
                          {"beta2", 0.99f},
                          {"l2_reg", 0.0f},
                      }},
        {"encoding", {
                         {"otype", "OneBlob"},
                         {"n_bins", 32},
                     }},
        {"network", {
                        {"otype", "FullyFusedMLP"},
                        {"n_neurons", 64},
                        {"n_hidden_layers", 4},
                        {"activation", "ReLU"},
                        {"output_activation", "None"},
                    }},
    };

    // load image
    std::filesystem::path filename(__FILE__);
    filename = filename.parent_path().parent_path() / "data" / "test.jpg";
    int width, height, nchannels;
    float *data = stbi_loadf(filename.c_str(), &width, &height, &nchannels, 4);
    tcnn::GPUMemory<float> image(width * height * 4);
    image.copy_from_host(data);
    stbi_image_free(data);

    // create a cuda texture object
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypePitch2D;
    resDesc.res.pitch2D.devPtr = image.data();
    resDesc.res.pitch2D.desc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
    resDesc.res.pitch2D.width = width;
    resDesc.res.pitch2D.height = height;
    resDesc.res.pitch2D.pitchInBytes = width * 4 * sizeof(float);

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.normalizedCoords = true;
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.addressMode[2] = cudaAddressModeClamp;

    cudaTextureObject_t texture;
    checkCudaErrors(cudaCreateTextureObject(&texture, &resDesc, &texDesc, nullptr));

    // train model
    const int n_input_dims = 2, n_output_dims = 3;
    const int batch_size = 1 << 16;
    auto model = tcnn::create_from_config(n_input_dims, n_output_dims, config);

    tcnn::GPUMatrix<float> training_batch_inputs(n_input_dims, batch_size);
    tcnn::GPUMatrix<float> training_batch_targets(n_output_dims, batch_size);

    tcnn::default_rng_t rng{0};

    auto start = std::chrono::steady_clock::now();
    const int n_training_steps = 50000;
    for (int i = 1; i <= n_training_steps; ++i)
    {
        tcnn::generate_random_uniform<float>(rng, batch_size * n_input_dims, training_batch_inputs.data());
        tcnn::linear_kernel(eval_image<n_output_dims>, 0, 0, batch_size, texture, training_batch_inputs.data(), training_batch_targets.data());

        auto ctx = model.trainer->training_step(training_batch_inputs, training_batch_targets);
        if (i % 5000 == 0)
        {
            float loss = model.trainer->loss(*ctx);
            std::cout << "  iteration=" << i << " loss=" << loss << std::endl;
        }
    }
    auto end = std::chrono::steady_clock::now();
    std::cout << "  training time: " << std::chrono::duration_cast<std::chrono::seconds>(end - start).count() << " s" << std::endl;

    // inference
    std::vector<float2> coords(width * height);
    for (int i = 0; i < width * height; ++i)
    {
        coords[i].x = ((i % width) + 0.5f) / (float)width;
        coords[i].y = ((i / width) + 0.5f) / (float)height;
    }
    tcnn::GPUMemory<float> coords_gpu(coords.size() * 2);
    coords_gpu.copy_from_host((float *)coords.data());
    tcnn::GPUMatrix<float> inference_inputs((float *)coords_gpu.data(), n_input_dims, width * height);
    tcnn::GPUMatrix<float> pred(n_output_dims, width * height);
    model.network->inference(inference_inputs, pred);

    std::vector<float3> pixels(width * height);
    checkCudaErrors(cudaMemcpy(pixels.data(), pred.data(), pixels.size() * sizeof(float3), cudaMemcpyDeviceToHost));
    std::vector<uchar3> img(width * height);
    for (int i = 0; i < width * height; ++i)
    {
        img[i].x = std::pow(std::clamp(pixels[i].x, 0.0f, 1.0f), 1.0f / 2.2f) * 255.0f;
        img[i].y = std::pow(std::clamp(pixels[i].y, 0.0f, 1.0f), 1.0f / 2.2f) * 255.0f;
        img[i].z = std::pow(std::clamp(pixels[i].z, 0.0f, 1.0f), 1.0f / 2.2f) * 255.0f;
    }
    std::filesystem::path outfile = filename.parent_path() / "out.jpg";
    int ret = stbi_write_jpg(outfile.c_str(), width, height, 3, img.data(), 100);
    if (ret == 0)
        std::cout << "failed to write image" << std::endl;

    tcnn::free_all_gpu_memory_arenas();

    std::cout << "Successfully tested TinyCudaNN" << std::endl;
}

void Matrix_test()
{
    // SquareMatrix<3> A = SquareMatrix<3>::Zero();
    // std::cout << Determinant(A) << std::endl;

    // auto t = Inverse(A);
    // if(t)
    //     std::cout << (*t)[0][0] << std::endl;
    // else
    //     std::cout << "inverse not possible" << std::endl;

    // SquareMatrix<3> B(1, 2, 3, 4, 3, 1, 7, 5, 2);
    // std::cout << Determinant(B) << std::endl;

    // auto t2 = Inverse(B);
    // if(t2)
    //     std::cout << (*t2)[0][0] << std::endl;
    // else
    //     std::cout << "inverse not possible" << std::endl;


    // curandState state;
    // curandStatePhilox4_32_10_t state2;
    // int seq = threadIdx.x;
    // curand_init(0, seq, 0, &state);
    // curand_init(0, seq, 0, &state2);
    // printf("%f %f\n", curand_uniform(&state), curand_uniform(&state));
    // auto t = curand_uniform4(&state2);
    // printf("%f %f %f %f\n", t.x, t.y, t.z, t.w);

    std::cout << Radians(90) << std::endl;

}

__global__ void film_test(float3* data, int w, int h)
{
    for(int i = 0; i < w; i++)
        for(int j = 0; j < h; j++)
            data[j * w + i] = make_float3(i / (float)w, j / (float)h, 0);
}

void Film_test()
{
    int w = 1024, h = 1024;
    Film film(w, h);
    film_test<<<1, 1>>>(film.getPtr(), w, h);
    checkCudaErrors(cudaDeviceSynchronize());
    std::filesystem::path filename(__FILE__);
    filename = filename.parent_path().parent_path() / "data" / "film_test.exr";
    film.saveExr(filename);
}

__global__ void device_test()
{
    printf("%f\n", Radians(90));
}


void Scene_test()
{
    std::filesystem::path filename(__FILE__);
    filename = filename.parent_path().parent_path() / "data" / "CornellBox" / "CornellBox-Sphere.obj";

    Model model(filename);
    std::cout << model.meshes.size() << " " << model.textures.size() << std::endl;
}


void optixApp_test()
{
    std::filesystem::path filename(__FILE__);
    filename = filename.parent_path().parent_path() / "data" / "CornellBox" / "CornellBox-Sphere.obj";

    Model model(filename);
    std::cout << model.meshes.size() << " " << model.textures.size() << std::endl;

    int w = 800, h = 600;
    OptixApp app(&model, w, h);
    std::vector<float4> pixels(w * h);

    bool useGUI = true;
    if(useGUI)
    {
        GUI gui(w, h);
        while(!gui.shouldClose())
        {
            // std::cout << "start rendering ..." << std::endl;
            app.render();
            // std::cout << "finished render" << std::endl;
            app.download(pixels.data());

            gui.run(pixels, w, h);
        }
    }
    else
    {
        std::cout << "start rendering ..." << std::endl;
        app.render();
        std::cout << "finished render" << std::endl;
        app.download(pixels.data());

        std::vector<uchar4> hhh;
        for(auto p : pixels)
            hhh.push_back(make_uchar4(p.x * 255.f, p.y * 255.f, p.z * 255.f, p.w * 255.f));

        std::filesystem::path outname(__FILE__);
        outname = outname.parent_path().parent_path() / "data" / "optixApp.jpg";
        stbi_flip_vertically_on_write(true);
        int ret = stbi_write_jpg(outname.c_str(), w, h, 4, (void *)hhh.data(), 1);
        if (ret == 0)
            std::cout << "Error writing file " << filename << std::endl;
    }
}

int main()
{
    // STB_test();
    // TinyExr_test();
    // OptiX_test();
    // TinyCudaNN_test();

    // Matrix_test();
    // device_test<<<1, 1>>>();
    // checkCudaErrors(cudaDeviceSynchronize());

    // Film_test();

    // Scene_test();

    optixApp_test();
    
    return 0;
}