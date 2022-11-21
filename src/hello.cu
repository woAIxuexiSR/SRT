#include <iostream>
#include "srt_math.h"
#include "gui.h"
#include <vector>
#include "helper_optix.h"
#include "optixRayTracer.h"
#include "definition.h"
#include "film.h"
#include "renderer.h"
#include "scene/camera.h"
#include "scene/material.h"
#include <type_traits>

__global__ void kernel()
{
}

template<class T>
void func(T t)
{
    std::cout << typeid(T).name() << " " << t << std::endl;
}

int main()
{
    // std::filesystem::path filename(__FILE__);
    // filename = filename.parent_path().parent_path() / "data" / "CornellBox" / "CornellBox-Original.obj";
    // // filename = filename.parent_path().parent_path() / "data" / "SimpleSphere" / "sphere.obj";
    // Renderer renderer(filename, 800, 600, false);
    // renderer.render();

    // kernel<<<1024, 600>>>();
    // checkCudaErrors(cudaDeviceSynchronize());

    return 0;
}