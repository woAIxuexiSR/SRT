#include <iostream>
#include "srt_math.h"
#include "gui.h"
#include <vector>
#include "helper_optix.h"
#include "optixRayTracer.h"
#include "integrator.h"
#include "definition.h"
#include "film.h"
#include "renderer.h"
#include "scene/camera.h"
#include "scene/material.h"
#include "scene/light.h"

void test_render()
{
    const int width = 1920, height = 1080;
    float aspect = (float)width / (float)height;

    std::filesystem::path currentPath(__FILE__);

    std::filesystem::path modelPath;
    modelPath = currentPath.parent_path().parent_path() / "data" / "CornellBox" / "CornellBox-Original.obj";
    // modelPath = currentPath.parent_path().parent_path() / "data" / "CornellBox" / "CornellBox-Glossy.obj";
    // modelPath = currentPath.parent_path().parent_path() / "data" / "CornellBox" / "CornellBox-Mirror.obj";
    // modelPath = currentPath.parent_path().parent_path() / "data" / "CornellBox" / "CornellBox-Water.obj";
    // modelPath = currentPath.parent_path().parent_path() / "data" / "SimpleSphere" / "sphere.obj";
    // modelPath = currentPath.parent_path().parent_path() / "data" / "sponza" / "sponza.obj";

    std::filesystem::path outPath;
    outPath = currentPath.parent_path().parent_path() / "data" / "out.exr";


    auto film = std::make_shared<Film>(width, height);
    auto model = std::make_shared<Model>(modelPath.string());
    auto camera = std::make_shared<Camera>(make_float3(0.0f, 1.0f, 0.5f), 5.0f, aspect);

    auto rayTracer = std::make_shared<PathTracer>(model.get());
    // auto rayTracer = std::make_shared<LightTracer>(model.get());
    // auto rayTracer = std::make_shared<BDPT>(model.get());
    // auto rayTracer = std::make_shared<DirectLight>(model.get());
    // rayTracer->setSamplesPerPixel(65536 * 4);
    // rayTracer->setSamplesPerPixel(4096);

    ImageRender imageRender(film, model, camera, rayTracer, outPath.string());
    TICK(time);
    imageRender.render();
    TOCK(time);

    // auto gui = std::make_shared<Gui>(width, height, camera);
    // InteractiveRender interactiveRender(film, model, camera, rayTracer, gui);
    // interactiveRender.render();

    // outPath = outPath.parent_path() / "test.mp4";
    // VideoRender videoRender(film, model, camera, rayTracer, outPath.string());
    // TICK(time);
    // videoRender.render();
    // TOCK(time);
}

__global__ void kernel(int num, Material** mat)
{
    for(int i = 0; i < num; i++)
    {
        mat[i] = new LambertianMaterial(make_float3(0.5f, 0.5f, 0.5f));
    }
}

__global__ void kernel2(Material* mat)
{
    float3 color = mat->getColor();
    printf("%f %f %f\n", color.x, color.y, color.z);
    MaterialType type = mat->getType();
    printf("%d\n", type);
    MaterialSample ms = mat->Sample(make_float3(1.0f, 1.0f, 0.0f), make_float3(0.0f, 1.0f, 0.0f), make_float2(0.3f, 0.6f),
        make_float3(0.2f, 0.4f, 0.5f), false);
    printf("%f %f %f %f\n", ms.f.x, ms.f.y, ms.f.z, ms.pdf);
}

__global__ void kernel3(Material* mat)
{
    delete mat;
}

int main()
{
    test_render();

    // GPUMemory<Material*> mat;
    // mat.resize(1);
    // kernel<<<1, 1>>>(1, mat.data());
    // checkCudaErrors(cudaDeviceSynchronize());

    // std::vector<Material*> mat_host;
    // mat_host.resize(1);
    // mat.copy_to_host(mat_host);

    // kernel2<<<1, 1>>>(mat_host[0]);
    // checkCudaErrors(cudaDeviceSynchronize());
    
    // kernel3<<<1, 1>>>(mat_host[0]);
    // checkCudaErrors(cudaDeviceSynchronize());

    return 0;
}