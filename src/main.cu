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

__host__ __device__ float3 func()
{
    return {0.1f, 0.2f, 0.3f};
}

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

__global__ void kernel()
{
    auto t = func();
    printf("%f %f %f\n", t.x, t.y, t.z);
}

int main()
{
    test_render();

    // kernel<<<1, 1>>>();
    // checkCudaErrors(cudaDeviceSynchronize());

    return 0;
}