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

__global__ void kernel()
{
    float2 sample = make_float2(3.766134e-01f, 3.841706e-09f);
    DiffuseMaterial mat(make_float3(0.5f, 0.5f, 0.5f));

    float3 wo = make_float3(0.555849f, -0.763161f, -0.329569f);
    float3 n = make_float3(0.0f, 0.0f, 1.0f);

    auto p = mat.Sample(-wo, n, sample);
    printf("p.wi = %e %e %e\n", p.wi.x, p.wi.y, p.wi.z);
    printf("p.pdf = %e\n", p.pdf);
}

int main()
{
    std::filesystem::path filename(__FILE__);
    filename = filename.parent_path().parent_path() / "data" / "CornellBox" / "CornellBox-Original.obj";
    // filename = filename.parent_path().parent_path() / "data" / "SimpleSphere" / "sphere.obj";
    Renderer renderer(filename, 800, 600, false);
    renderer.render();

    // kernel<<<1, 1>>>();
    // checkCudaErrors(cudaDeviceSynchronize());

    return 0;
}