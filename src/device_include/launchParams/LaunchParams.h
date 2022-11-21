#pragma once

#include "scene/camera.h"

template <class T>
struct LaunchParams
{
    int frameId;
    int width, height;
    OptixTraversableHandle traversable;

    Camera camera;
    float4* colorBuffer;
    T extraData;

    LaunchParams() {}
    LaunchParams(int _w, int _h, OptixTraversableHandle _t) : frameId(0), width(_w), height(_h), traversable(_t) {}
};