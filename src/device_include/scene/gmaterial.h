#pragma once

#include "my_math.h"
#include "bxdf.h"

class GMaterial
{
public:
    BxDF bxdf;
    float3 base_color{ 0.0f, 0.0f, 0.0f };
    float3 emission_color{ 0.0f, 0.0f, 0.0f };
    float intensity{ 0.0f };

    cudaTextureObject_t color_tex{ 0 };
    cudaTextureObject_t normal_tex{ 0 };

public:
    __host__ __device__ GMaterial() {}

    __host__ __device__ bool is_emissive() const { return intensity > 0.0f; }
    __host__ __device__ bool is_specular() const
    {
        return (bxdf.type == BxDF::Type::Dielectric)
            || (bxdf.type == BxDF::Type::Disney && bxdf.metallic >= 0.99f && bxdf.roughness <= 0.01f)
            || (bxdf.type == BxDF::Type::Disney && bxdf.specTrans >= 0.99f && bxdf.roughness <= 0.01f && bxdf.clearcoatGloss <= 0.01f);
    }
    __host__ __device__ bool is_transmissive() const
    {
        return (bxdf.type == BxDF::Type::Dielectric)
            || (bxdf.type == BxDF::Type::Disney && bxdf.metallic < 1.0f && bxdf.specTrans > 0.0f);
    }

    /* functions to get surface infomation */

    __device__ float3 emission(float2 texcoord) const
    {
        float3 e = emission_color;
        if (color_tex)
            e = make_float3(tex2D<float4>(color_tex, texcoord.x, texcoord.y));
        return e * intensity;
    }

    __device__ float3 shading_normal(float3 normal, float2 texcoord) const
    {
        if (normal_tex)
            return normalize(make_float3(tex2D<float4>(normal_tex, texcoord.x, texcoord.y)));
        return normal;
    }

    __device__ float3 surface_color(float2 texcoord) const
    {
        if (color_tex)
            return make_float3(tex2D<float4>(color_tex, texcoord.x, texcoord.y));
        return base_color;
    }

    /* onb build from shading normal and tangent, color is surface color */

    __device__ float3 eval(float3 wi, float3 wo, const Onb& onb, float3 color) const
    {
        float3 wi_local = onb.to_local(wi);
        float3 wo_local = onb.to_local(wo);

        bool inner = false;
        if (wo_local.z <= 0.0f)  // reverse the three axes of onb
        {
            wo_local = -wo_local;
            wi_local = -wi_local;
            inner = true;
        }
        return bxdf.eval(wi_local, wo_local, color, inner);
    }

    __device__ BxDFSample sample(float3 wo, float2 rnd, const Onb& onb, float3 color) const
    {
        float3 wo_local = onb.to_local(wo);

        bool inner = false;
        if (wo_local.z <= 0.0f)
        {
            wo_local = -wo_local;
            inner = true;
        }

        BxDFSample bxdf_sample = bxdf.sample(wo_local, rnd, color, inner);
        if (inner) bxdf_sample.wi = -bxdf_sample.wi;
        bxdf_sample.wi = onb.to_world(bxdf_sample.wi);
        return bxdf_sample;
    }

    __device__ float sample_pdf(float3 wi, float3 wo, const Onb& onb, float3 color) const
    {
        float3 wi_local = onb.to_local(wi);
        float3 wo_local = onb.to_local(wo);

        bool inner = false;
        if (wo_local.z <= 0.0f)
        {
            wo_local = -wo_local;
            wi_local = -wi_local;
            inner = true;
        }
        return bxdf.sample_pdf(wi_local, wo_local, color, inner);
    }
};