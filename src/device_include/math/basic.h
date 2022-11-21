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

__host__ __device__ inline float FrDielectric(float cosTheta_i, float eta)
{
    if (cosTheta_i < 0)
    {
        cosTheta_i = -cosTheta_i;
        eta = 1 / eta;
    }
    float sin2Theta_i = 1 - cosTheta_i * cosTheta_i;
    float sin2Theta_t = sin2Theta_i / (eta * eta);
    if (sin2Theta_t >= 1)
        return 1.0f;
    float cosTheta_t = sqrt(1 - sin2Theta_t);

    float r_parl = (eta * cosTheta_i - cosTheta_t) / (eta * cosTheta_i + cosTheta_t);
    float r_perp = (cosTheta_i - eta * cosTheta_t) / (cosTheta_i + eta * cosTheta_t);
    return (r_parl * r_parl + r_perp * r_perp) / 2.0f;
}

__host__ __device__ inline float FrSchlickDielectric(float cosTheta_i, float eta)
{
    if (cosTheta_i < 0)
    {
        cosTheta_i = -cosTheta_i;
        eta = 1 / eta;
    }
    float sin2Theta_i = 1 - cosTheta_i * cosTheta_i;
    float sin2Theta_t = sin2Theta_i / (eta * eta);
    if (sin2Theta_t >= 1)
        return 1.0f;

    float R0 = (eta - 1) / (eta + 1);
    R0 = R0 * R0;
    return R0 + (1 - R0) * pow(1 - cosTheta_i, 5.0f);
}