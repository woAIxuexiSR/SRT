#pragma once

#include "scene/camera.h"
#include "scene/light.h"
#include "bdptParams.h"

template <class T>
struct LaunchParams
{
    int SPP;
    int MAX_DEPTH;
    float3 background;

    int frameId;
    int width, height;
    OptixTraversableHandle traversable;

    Camera camera;
    Light light;

    float4* colorBuffer;
    T extraData;

    LaunchParams() {}
    LaunchParams(int _w, int _h, OptixTraversableHandle _t) : frameId(0), width(_w), height(_h), traversable(_t) {}
};