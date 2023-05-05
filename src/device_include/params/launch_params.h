#pragma once

#include "scene/camera.h"
#include "scene/light.h"
#include "scene/material.h"

#include "helper_optix.h"
#include "params_data.h"

template<class T>
class LaunchParams
{
public:
    int seed;
    int width, height;
    OptixTraversableHandle traversable;

    Camera camera;
    float4* pixels;

    T extra;

public:
    LaunchParams() = default;
    LaunchParams(int _s, int _w, int _h, OptixTraversableHandle _t, Camera _c, float4* _p, T _e) :
        seed(_s), width(_w), height(_h), traversable(_t), camera(_c), pixels(_p), extra(_e) {}
};

using PathTracerParams = LaunchParams<PathTracerData>;