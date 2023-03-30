#pragma once

#include "scene/camera.h"
#include "scene/light.h"
#include "scene/material.h"

#include "helper_optix.h"
#include "bdpt_params.h"

#define MAX_DEPTH 16

template <class T>
class LaunchParams
{
public:
    int spp;
    float3 background;
    int frame{ 0 };

    int width, height;
    OptixTraversableHandle traversable;

    Camera camera;
    Light light;

    float4* buffer;
    T extra;

public:
    LaunchParams() {}
};


struct HitInfo
{
    bool hit;
    float3 pos;
    float3 normal;
    const Material* mat;
    float3 color;
};


enum class SimpleShadeType
{
    Ambient,
    BaseColor,
    Normal,
    Depth,
};