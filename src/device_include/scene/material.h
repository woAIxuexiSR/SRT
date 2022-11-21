#pragma once

#include <cuda_runtime.h>
#include "srt_math.h"

// wi: incoming direction  => from light to surface (sample)
// wo: outgoing direction  => from surface to camera

enum class MaterialType
{
    Diffuse = 1,
    Specular = 2,
    Glossy = 4,
    Transmission = 8,
    ALL = Diffuse | Specular | Glossy | Transmission
};

class MaterialSample
{
public:
    MaterialType type;  // material type
    float3 f;           // BRDF
    float3 wi;          // incoming direction
    float pdf;          // pdf of wi

    __device__ MaterialSample() {}
    __device__ MaterialSample(MaterialType _t, float3 _f, float3 _wi, float _pdf)
        : type(_t), f(_f), wi(_wi), pdf(_pdf) {}
};

class DiffuseMaterial
{
private:
    float3 m_color;

public:
    __host__ __device__ DiffuseMaterial(float3 color = make_float3(1.0f)) : m_color(color) {}

    __host__ __device__ float3 getColor() const { return m_color; }

    // return brdf * cosTheta
    __device__ float3 Eval(float3 wi, float3 wo, float3 n) const
    {
        if (dot(wi, n) <= 0.0f || dot(wo, n) <= 0.0f)
            return make_float3(0.0f);
        return m_color * M_1_PI * dot(wi, n);
    }

    __device__ MaterialSample Sample(float3 wo, float3 n, float2 sample) const
    {
        float3 v = CosineSampleHemiSphere(sample);
        Onb onb(n);
        float3 wi = onb(v);
        return MaterialSample(MaterialType::Diffuse, Eval(wi, wo, n), wi, Pdf(wi, wo, n));
    }

    __device__ float Pdf(float3 wi, float3 wo, float3 n) const
    {
        float cosTheta = dot(wi, n);
        return CosineHemiSpherePdf(cosTheta);
    }
};

// class GlossyMaterial
// {
// private:
//     float3 m_color;
//     float m_roughness;
// };

// class SpecularMaterial
// {
// };

// class RefractiveMaterial
// {
// };


// class DisneyMaterial
// {
// };


// class DielectricBxDF : public BxDF
// {
// private:
//     float eta;
//     float3 m_color;

// public:
//     DielectricBxDF() {}
//     DielectricBxDF(float _e, float3 _c) : eta(_e), m_color(_c) {}

//     __device__ float3 Eval(float3 wi, float3 wo) const override
//     {
//         return make_float3(0.0f);
//     }

//     __device__ BxDFSample Sample(float3 wo, float2 sample) const override
//     {
//         float Fr = FrDielectric(wo.z, eta);
//         if(sample.x <= Fr)
//         {
//             float3 wi = make_float3(-wo.x, -wo.y, wo.z);
//             return BxDFSample(BxDFType::Specular, m_color, wi, Fr);
//         }
//         else
//         {
//             float sin2Theta_t = (1.0 - wo.z * wo.z) / (eta * eta);
//             float cosTheta_t = sqrt(max(0.0f, 1.0f - sin2Theta_t));
//             float3 wi = -wo / eta + (wo.z / eta - cosTheta_t) * make_float3(0.0f, 0.0f, 1.0f);
//             return BxDFSample(BxDFType::Transmission, m_color, wi, 1.0f - Fr);
//         }
//     }
// };