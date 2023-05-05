#pragma once

#include "scene/camera.h"
#include "scene/light.h"
#include "scene/material.h"

class PathTracerData
{
public:
    int spp;
    int max_depth;

    Light light;
    float3 background;

public:
    PathTracerData() = default;
    PathTracerData(int _s, int _d, Light _l, float3 _b)
        : spp(_s), max_depth(_d), light(_l), background(_b) {}
};