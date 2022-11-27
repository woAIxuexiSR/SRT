#pragma once

#include <cuda_runtime.h>
#include "srt_math.h"

// wi: incoming direction  => from surface to light (sample)
// wo: outgoing direction  => from surface to camera

enum class MaterialType
{
    Lambertian,
    Emissive,
    Disney,
};

class MaterialSample
{
public:
    MaterialType type;  // material type
    float f;           // BRDF
    float3 wi;          // incoming direction
    float pdf;          // pdf of wi

    __device__ MaterialSample() {}
    __device__ MaterialSample(MaterialType _t, float _f, float3 _wi, float _pdf)
        : type(_t), f(_f), wi(_wi), pdf(_pdf) {}
};

class Material
{
private:
    MaterialType type;

public:
    __host__ __device__ Material() : type(MaterialType::Lambertian) {}
    __host__ __device__ Material(MaterialType _t) : type(_t) {}

    __host__ __device__ void setType(MaterialType _t) { type = _t; }

    __host__ __device__ bool isLight() const { return type == MaterialType::Emissive; }

    __host__ __device__ bool isSpecular() const { return false; }

    __device__ float Eval(float3 wi, float3 wo, float3 n) const
    {
        if (type == MaterialType::Lambertian)
        {
            if (dot(wi, n) <= 0.0f || dot(wo, n) <= 0.0f)
                return 0.0f;
            return M_1_PI * dot(wi, n);
        }
        return 0.0f;
    }

    __device__ MaterialSample Sample(float3 wo, float3 n, float2 sample) const
    {
        if (type == MaterialType::Lambertian)
        {
            float3 v = CosineSampleHemiSphere(sample);
            Onb onb(n);
            float3 wi = onb(v);
            return MaterialSample(type, Eval(wi, wo, n), wi, Pdf(wi, wo, n));
        }
        return MaterialSample();
    }

    __device__ float Pdf(float3 wi, float3 wo, float3 n) const
    {
        if (type == MaterialType::Lambertian)
        {
            float cosTheta = dot(wi, n);
            return CosineHemiSpherePdf(cosTheta);
        }
        return 0.0f;
    }
};