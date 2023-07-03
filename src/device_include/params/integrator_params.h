#pragma once

#include "helper_optix.h"
#include "scene/camera.h"
#include "scene/light.h"
#include "scene/gmaterial.h"


class PathTracerParams
{
public:
    int seed;
    int width, height;
    OptixTraversableHandle traversable;

    Camera camera;
    Light* light;
    float4* pixels;

    int samples_per_pixel{ 1 };
    int max_depth{ 16 };
    int rr_depth{ 4 };
    bool use_nee{ true };
    bool use_mis{ true };
};