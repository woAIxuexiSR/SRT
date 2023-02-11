#pragma once

#include <cuda_runtime.h>
#include <thrust/pair.h>
#include "helper_math.h"

__host__ __device__ inline float Radians(float deg)
{
    return deg * M_PI / 180.0f;
}

// pack pointer
__host__ __device__ inline thrust::pair<unsigned, unsigned> packPointer(void* ptr)
{
    unsigned long long p = reinterpret_cast<unsigned long long>(ptr);
    return thrust::make_pair(static_cast<unsigned>(p >> 32), static_cast<unsigned>(p & 0xffffffff));
}

__host__ __device__ inline void* unpackPointer(unsigned i0, unsigned i1)
{
    unsigned long long p = static_cast<unsigned long long>(i0) << 32 | i1;
    return reinterpret_cast<void*>(p);
}


// sample
__host__ __device__ inline float3 UniformSampleHemiSphere(float2 sample)
{
    float z = sample.x;
    float r = sqrt(1.0f - z * z);
    float phi = 2.0f * M_PI * sample.y;
    return make_float3(r * cos(phi), r * sin(phi), z);
}

__host__ __device__ inline float UniformHemiSpherePdf()
{
    return 0.5f * M_1_PI;
}

__host__ __device__ inline float3 UniformSampleSphere(float2 sample)
{
    float z = 1.0f - 2.0f * sample.x;
    float r = sqrt(1.0f - z * z);
    float phi = 2.0f * M_PI * sample.y;
    return make_float3(r * cos(phi), r * sin(phi), z);
}

__host__ __device__ inline float UniformSpherePdf()
{
    return 0.25f * M_1_PI;
}

__host__ __device__ inline float2 UniformSampleDisk(float2 sample)
{
    float r = sqrt(sample.x);
    float theta = 2.0f * M_PI * sample.y;
    return r * make_float2(cos(theta), sin(theta));
}

__host__ __device__ inline float2 ConcentricSampleDisk(float2 sample)
{
    float2 uOffset = 2.0f * sample - 1.0f;
    if (uOffset.x == 0 && uOffset.y == 0)
        return make_float2(0.0f, 0.0f);
    float theta, r;
    if (fabs(uOffset.x) > fabs(uOffset.y))
    {
        r = uOffset.x;
        theta = M_PI_4 * (uOffset.y / uOffset.x);
    }
    else
    {
        r = uOffset.y;
        theta = M_PI_2 - M_PI_4 * (uOffset.x / uOffset.y);
    }
    return r * make_float2(cos(theta), sin(theta));
}

__host__ __device__ inline float UniformDiskPdf()
{
    return M_1_PI;
}

__host__ __device__ inline float3 CosineSampleHemiSphere(float2 sample)
{
    float2 d = ConcentricSampleDisk(sample);
    float z = sqrt(fmax(0.0f, 1.0f - d.x * d.x - d.y * d.y));
    return make_float3(d.x, d.y, z);
}

__host__ __device__ inline float CosineHemiSpherePdf(float cosTheta)
{
    return cosTheta * M_1_PI;
}

__host__ __device__ inline float3 UniformSampleCone(float2 sample, float cosThetaMax)
{
    float cosTheta = (1.0f - sample.x) + sample.x * cosThetaMax;
    float sinTheta = sqrt(1.0f - cosTheta * cosTheta);
    float phi = 2.0f * M_PI * sample.y;
    return make_float3(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta);
}

__host__ __device__ inline float2 UniformSampleTriangle(float2 sample)
{
    float sx = sqrt(sample.x);
    return make_float2(1.0f - sx, sample.y * sx);
}


// disney material helper funcitons
__host__ __device__ inline float Fresnel(float cosThetaI, float cosThetaT, float eta)
{
    float rParl = ((eta * cosThetaI) - cosThetaT) / ((eta * cosThetaI) + cosThetaT);
    float rPerp = ((cosThetaI) - (eta * cosThetaT)) / ((cosThetaI) + (eta * cosThetaT));
    return (rParl * rParl + rPerp * rPerp) * 0.5f;
}

__host__ __device__ inline float SchlickFresnel(float u)
{
    float m = clamp(1.0f - u, 0.0f, 1.0f);
    float m2 = m * m;
    return m2 * m2 * m;
}

__host__ __device__ inline float GTR1(float NdotH, float a)
{
    if (a >= 1.0f) return (1.0f / M_PI);
    float a2 = a * a;
    float t = 1.0f + (a2 - 1.0f) * NdotH * NdotH;
    return (a2 - 1.0f) / (M_PI * log(a2) * t);
}

__host__ __device__ inline float GTR2(float NdotH, float a)
{
    float a2 = a * a;
    float t = 1.0f + (a2 - 1.0f) * NdotH * NdotH;
    return a2 / (M_PI * t * t);
}

__host__ __device__ inline float GTR2_aniso(float NdotH, float HdotX, float HdotY, float ax, float ay)
{
    float t = (HdotX * HdotX) / (ax * ax) + (HdotY * HdotY) / (ay * ay) + NdotH * NdotH;
    return 1.0f / (M_PI * ax * ay * t * t);
}

__host__ __device__ inline float smithG_GGX(float NdotV, float alphaG)
{
    float a = alphaG * alphaG;
    float b = NdotV * NdotV;
    return 1.0f / (NdotV + sqrt(a + b - a * b));
}

__host__ __device__ inline float smithG_GGX_aniso(float NdotV, float VdotX, float VdotY, float ax, float ay)
{
    float t = (VdotX * VdotX) / (ax * ax) + (VdotY * VdotY) / (ay * ay) + NdotV * NdotV;
    return 1.0f / (NdotV + sqrt(t));
}