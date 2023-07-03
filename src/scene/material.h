#pragma once

#include "definition.h"
#include "my_math.h"
#include "scene/gmaterial.h"

class Material
{
public:
    string name;

    BxDF bxdf;
    float3 base_color{ 0.0f, 0.0f, 0.0f };
    float3 emission_color{ 0.0f, 0.0f, 0.0f };
    float intensity{ 0.0f };

    int color_tex_id{ -1 };
    int normal_tex_id{ -1 };

public:
    Material() {}

    void set_type(BxDF::Type type);
    void set_property(const string& str, float value);
};