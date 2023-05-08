#pragma once

#include "params/launch_params.h"

struct HitInfo
{
    bool hit;
    
    float3 pos;
    float3 normal;
    bool inner;
    
    const Material* mat;
    float3 color;

    int light_id;
};