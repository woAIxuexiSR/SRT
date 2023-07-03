#pragma once

#include "params/integrator_params.h"
#include "params/npr_params.h"
#include "params/wavefront_params.h"

class HitInfo
{
public:
    bool hit;

    float3 pos;
    float3 normal;
    float2 texcoord;
    Onb onb;    // build from shading normal
    float3 color;
    const GMaterial* mat;

    bool inner;
    int light_id;
};