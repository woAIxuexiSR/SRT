#pragma once

#include "helper_optix.h"
#include "scene/camera.h"
#include "scene/light.h"
#include "scene/gmaterial.h"
#include "scene/gmesh.h"


class SimpleParams
{
public:
    enum class Type { Depth, Normal, BaseColor, Ambient, FaceOrientation };

    int seed;
    int width, height;
    OptixTraversableHandle traversable;

    Camera camera;
    float4* pixels;

    Type type;
    int samples_per_pixel{ 16 };
    float min_depth{ 0.0f };
    float max_depth{ 1000.0f };
};


class NRLHSParams
{
public:
    int seed;
    int width, height;
    OptixTraversableHandle traversable;

    Camera camera;
    float* inference_buffer;
    bool* mask;
    float3* base;
};

class NRRHSParams
{
public:
    int seed;
    int width, height;
    OptixTraversableHandle traversable;

    Camera camera;
    Light* light;

    float* inference_buffer;
    bool* mask;
    float3* weight;
    float3* base;
};