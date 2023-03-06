#pragma once

#include "scene/camera.h"
#include "scene/light.h"
#include "scene/material.h"

#include "bdptParams.h"

#define MAX_DEPTH 16

template <class T>
class LaunchParams
{
public:
    int samplesPerPixel{ 128 };
    float3 background{ 0.0f, 0.0f, 0.0f };
    int frameId{ 0 };

    int width, height;
    OptixTraversableHandle traversable;

    Camera camera;
    Light light;

    float4* colorBuffer;
    T extraData;

public:
    LaunchParams() {}
};


struct HitInfo
{
    bool isHit;

    float3 hitPos;
    float3 hitNormal;
    const Material* mat;
    float3 color;
};