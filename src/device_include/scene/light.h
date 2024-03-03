#pragma once

#include "my_math.h"
#include "gmesh.h"

class LightSample
{
public:
    float3 pos;
    float3 normal;
    float3 emission{ 0.0f, 0.0f, 0.0f };
    float pdf{ 0.0f };
};

class EnvironmentLight
{
public:
    enum class Type { Constant, UVMap };

    Type type{ Type::Constant };
    float3 emission_color{ 0.0f, 0.0f, 0.0f };
    cudaTextureObject_t texture{ 0 };

public:
    EnvironmentLight() {}

    __device__ float3 emission(float3 dir) const
    {
        switch (type)
        {
        case Type::Constant:
            return emission_color;
        case Type::UVMap:
        {
            // world space y axis is up, math space z axis is up
            float3 math_space_dir = make_float3(-dir.x, dir.z, dir.y);
            float2 phi_theta = cartesian_to_spherical_uv(math_space_dir);
            float2 uv = make_float2(phi_theta.x * 0.5f * (float)M_1_PI + 0.5f, phi_theta.y * (float)M_1_PI);
            return make_float3(tex2D<float4>(texture, uv.x, uv.y));
        }
        default:
            return { 0.0f, 0.0f, 0.0f };
        }
    }
};

// many area lights and one environment light
class Light
{
public:
    int num{ 0 };       // number of area lights
    GInstance* lights{ nullptr };
    float weight_sum{ 0.0f };
    float* weight_cdf{ nullptr };   // area * intensity cdf

    EnvironmentLight* env_light{ nullptr };

public:
    Light() {}

    __device__ LightSample sample(float2 rnd) const
    {
        int idx = binary_search(weight_cdf, num, rnd.x);

        int lower = (idx == 0) ? 0 : weight_cdf[idx - 1];
        rnd.x = (rnd.x - lower) / (weight_cdf[idx] - lower);
        SurfaceSample ms = lights[idx].sample(rnd);

        return { ms.pos, ms.normal, ms.mat->emission(ms.texcoord), sample_pdf(idx) };
    }

    // sample proportional to area * intensity
    __device__ float sample_pdf(int idx) const
    {
        return lights[idx].mesh->material->intensity / weight_sum;
    }

    __device__ float3 environment_emission(float3 dir) const
    {
        if (env_light)
            return env_light->emission(dir);
        return { 0.0f, 0.0f, 0.0f };
    }
};