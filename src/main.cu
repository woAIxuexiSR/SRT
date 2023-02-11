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
#include "scene/light.h"

__global__ void kernel()
{
}

float3 test(float3 wo, float3 n, float2 sample)
{
    float eta = 1.0f / 1.5f, cosi = dot(wo, n);
    if (cosi <= 0.0f)
    {
        eta = 1.0f / eta;
        cosi = -cosi;
        n = -n;
    }

    float sint = eta * sqrt(max(0.0f, 1.0f - cosi * cosi));
    float cost = sqrt(max(0.0f, 1.0f - sint * sint));
    float reflectRatio = sint >= 1.0f ? 1.0f : Fresnel(cosi, cost, eta);
    std::cout << sint << " " << cost << std::endl;

    float3 rperp = -eta * (wo - n * cosi);
    float3 rparl = -sqrt(max(0.0f, 1.0f - dot(rperp, rperp))) * n;
    return normalize(rperp + rparl);

    // float3 wi;
    // if (sample.x <= reflectRatio)
    //     wi = 2.0f * cosi * n - wo;
    // else
    // {
    //     float3 rperp = -eta * (wo - n * cosi);
    //     float3 rparl = -sqrt(max(0.0f, 1.0f - dot(rperp, rperp))) * n;
    //     wi = normalize(rperp + rparl);
    // }

    // return wi;
}

int main()
{
    std::filesystem::path filename(__FILE__);
    // filename = filename.parent_path().parent_path() / "data" / "CornellBox" / "CornellBox-Original.obj";
    filename = filename.parent_path().parent_path() / "data" / "CornellBox" / "CornellBox-Glossy.obj";
    // filename = filename.parent_path().parent_path() / "data" / "CornellBox" / "CornellBox-Mirror.obj";
    // filename = filename.parent_path().parent_path() / "data" / "CornellBox" / "CornellBox-Water.obj";
    // filename = filename.parent_path().parent_path() / "data" / "SimpleSphere" / "sphere.obj";
    // filename = filename.parent_path().parent_path() / "data" / "sponza" / "sponza.obj";
    Renderer renderer(filename, 800, 600, false);
    renderer.render();

    // kernel<<<1, 1>>>();
    // checkCudaErrors(cudaDeviceSynchronize());

    // float3 wo = normalize(make_float3(-1.f, 2.f, 0.0f));
    // float3 n = normalize(make_float3(0.0f, 1.0f, 0.0f));
    // float2 sample = make_float2(0.5f, 0.5f);
    // float3 wi = test(wo, n, sample);
    // std::cout << wi.x << " " << wi.y << " " << wi.z << std::endl;

    return 0;
}