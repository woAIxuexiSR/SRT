#pragma once

#include <cuda_runtime.h>

#ifdef _MSC_VER
#include "corecrt_math_defines.h"
#endif

#include "helper_math.h"

/* basic math */

__host__ __device__ inline float Radians(float deg)
{
    return deg * M_PI / 180.0f;
}

__host__ __device__ inline float Degrees(float rad)
{
    return rad * 180.0f / M_PI;
}

// normalized vector to spherical uv coordinate,  [-pi, pi] x [0, pi]
__host__ __device__ inline float2 cartesian_to_spherical_uv(float3 p)
{
    float phi = atan2(p.y, p.x);
    float theta = acos(p.z);
    return make_float2(phi, theta);
}

// [phi, theta] to normalized vector
__host__ __device__ inline float3 spherical_uv_to_cartesian(float2 uv)
{
    float sin_theta = sin(uv.y), cos_theta = cos(uv.y);
    return make_float3(cos(uv.x) * sin_theta, sin(uv.x) * sin_theta, cos_theta);
}

// pack pointer
__host__ __device__ inline uint2 pack_pointer(void* ptr)
{
    unsigned long long p = reinterpret_cast<unsigned long long>(ptr);
    return make_uint2(static_cast<unsigned>(p >> 32), static_cast<unsigned>(p & 0xffffffff));
}

__host__ __device__ inline void* unpack_pointer(unsigned i0, unsigned i1)
{
    unsigned long long p = static_cast<unsigned long long>(i0) << 32 | i1;
    return reinterpret_cast<void*>(p);
}


/* sample */

__host__ __device__ inline float3 uniform_sample_hemisphere(float2 sample)
{
    float z = sample.x;
    float r = sqrt(1.0f - z * z);
    float phi = 2.0f * M_PI * sample.y;
    return make_float3(r * cos(phi), r * sin(phi), z);
}

__host__ __device__ inline float uniform_hemisphere_pdf()
{
    return 0.5f * M_1_PI;
}

__host__ __device__ inline float3 uniform_sample_sphere(float2 sample)
{
    float z = 1.0f - 2.0f * sample.x;
    float r = sqrt(1.0f - z * z);
    float phi = 2.0f * M_PI * sample.y;
    return make_float3(r * cos(phi), r * sin(phi), z);
}

__host__ __device__ inline float uniform_sphere_pdf()
{
    return 0.25f * M_1_PI;
}

__host__ __device__ inline float2 uniform_sample_disk(float2 sample)
{
    float r = sqrt(sample.x);
    float theta = 2.0f * M_PI * sample.y;
    return r * make_float2(cos(theta), sin(theta));
}

__host__ __device__ inline float2 concentric_sample_disk(float2 sample)
{
    float2 offset = 2.0f * sample - 1.0f;
    if (offset.x == 0 && offset.y == 0)
        return make_float2(0.0f, 0.0f);
    float theta, r;
    if (fabs(offset.x) > fabs(offset.y))
    {
        r = offset.x;
        theta = M_PI_4 * (offset.y / offset.x);
    }
    else
    {
        r = offset.y;
        theta = M_PI_2 - M_PI_4 * (offset.x / offset.y);
    }
    return r * make_float2(cos(theta), sin(theta));
}

__host__ __device__ inline float uniform_disk_pdf()
{
    return M_1_PI;
}

__host__ __device__ inline float3 cosine_sample_hemisphere(float2 sample)
{
    float2 d = concentric_sample_disk(sample);
    float z = sqrt(fmax(0.0f, 1.0f - d.x * d.x - d.y * d.y));
    return make_float3(d.x, d.y, z);
}

__host__ __device__ inline float cosine_hemisphere_pdf(float cos_theta)
{
    return cos_theta * M_1_PI;
}

__host__ __device__ inline float2 uniform_sample_triangle(float2 sample)
{
    float sx = sqrt(sample.x);
    return make_float2(1.0f - sx, sample.y * sx);
}


/* disney material helper functions */


__host__ __device__ inline float Fresnel(float cos_theta_i, float cos_theta_t, float eta)
{
    float r_parl = (eta * cos_theta_i - cos_theta_t) / (eta * cos_theta_i + cos_theta_t);
    float r_perp = (cos_theta_i - eta * cos_theta_t) / (cos_theta_i + eta * cos_theta_t);
    return (r_parl * r_parl + r_perp * r_perp) * 0.5f;
}

__host__ __device__ inline float schlick_fresnel(float cos_theta, float eta = 1.0f)
{
    float r0 = (eta - 1.0f) / (eta + 1.0f);
    r0 = r0 * r0;
    float m = clamp(1.0f - cos_theta, 0.0f, 1.0f);
    return r0 + (1.0f - r0) * m * m * m * m * m;
}

__host__ __device__ inline float fresnel_mix(float metallic, float eta, float cos_theta)
{
    return lerp(schlick_fresnel(cos_theta, eta), schlick_fresnel(cos_theta), metallic);
}

__host__ __device__ inline float Luminance(float3 color)
{
    return dot(color, make_float3(0.2126f, 0.7152f, 0.0722f));
}

__host__ __device__ inline float GTR1(float n_h, float a)
{
    if (a >= 1.0f) return M_1_PI;
    float a2 = a * a;
    float t = 1.0f + (a2 - 1.0f) * n_h * n_h;
    return (a2 - 1.0f) / (M_PI * log(a2) * t);
}

__host__ __device__ inline float GTR2(float n_h, float a)
{
    float a2 = a * a;
    float t = 1.0f + (a2 - 1.0f) * n_h * n_h;
    return a2 / (M_PI * t * t);
}

__host__ __device__ inline float smithG_GGX(float n_v, float alphaG)
{
    float a = alphaG * alphaG;
    float b = n_v * n_v;
    return (2.0f * n_v) / (n_v + sqrt(a + b - a * b));
}

__host__ __device__ inline float3 sample_GTR1(float a, float2 sample)
{
    if (a >= 1.0f) return cosine_sample_hemisphere(sample);
    float a2 = a * a;
    float phi = 2.0f * M_PI * sample.x;
    float cos_theta = sqrt((1.0f - pow(a2, 1.0f - sample.y)) / (1.0f - a2));
    float sin_theta = clamp(sqrt(1.0f - (cos_theta * cos_theta)), 0.0f, 1.0f);
    return make_float3(cos(phi) * sin_theta, sin(phi) * sin_theta, cos_theta);
}

__host__ __device__ inline float3 sample_GTR2(float a, float2 sample)
{
    float a2 = a * a;
    float phi = 2.0f * M_PI * sample.x;
    float sin_theta = sqrt(a2 / (1.0f / sample.y - 1.0f + a2));
    float cos_theta = clamp(sqrt(1.0f - (sin_theta * sin_theta)), 0.0f, 1.0f);
    return make_float3(cos(phi) * sin_theta, sin(phi) * sin_theta, cos_theta);
}

__host__ __device__ inline float3 refract(float3 i, float3 n, float eta)
{
    float n_i = dot(n, -i);
    float k = 1.0f - eta * eta * (1.0f - n_i * n_i);
    if (k < 0.0f) return make_float3(0.0f);
    return eta * i + (eta * n_i - sqrt(k)) * n;
}