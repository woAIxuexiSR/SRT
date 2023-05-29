#pragma once

#include "params/integrator_params.h"
#include "params/npr_params.h"
#include "params/wavefront_params.h"

struct HitInfo
{
    bool hit;

    float3 pos, normal, color;
    const Material* mat;

    bool inner;
    int light_id;
};