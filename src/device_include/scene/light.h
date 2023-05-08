#pragma once

#include <cuda_runtime.h>
#include "my_math.h"

class LightSample
{
public:
    float3 pos;
    float3 normal;
    float3 emission;
    float pdf;

    __host__ __device__ LightSample() {}
    __host__ __device__ LightSample(float3 _p, float3 _n, float3 _e, float _pdf) : pos(_p), normal(_n), emission(_e), pdf(_pdf) {}
};


class DiffuseAreaLight
{
public:
    int face_num;
    float3* vertices;
    uint3* indices;
    float3* normals;
    float2* texcoords;
    cudaTextureObject_t texture;

    float3 emission_color;
    float intensity;

    float* areas;
    float area_sum;

public:
    __host__ __device__ DiffuseAreaLight() {}

    __device__ LightSample sample(float2 rnd)
    {
        float s = rnd.x * area_sum;
        float sum = 0.0f;
        for (int i = 0; i < face_num; i++)
        {
            if (s <= sum + areas[i] || i == face_num - 1)
            {
                rnd.x = clamp((s - sum) / areas[i], 0.0f, 1.0f);
                float2 p = uniform_sample_triangle(rnd);
                uint3& index = indices[i];

                float3 pos = vertices[index.x] * (1.0f - p.x - p.y) + vertices[index.y] * p.x + vertices[index.z] * p.y;

                float3 normal;
                if (normals)
                    normal = normals[index.x] * (1.0f - p.x - p.y) + normals[index.y] * p.x + normals[index.z] * p.y;
                else
                    normal = cross(vertices[index.y] - vertices[index.x], vertices[index.z] - vertices[index.x]);

                float3 emission = emission_color;
                if (texcoords)
                {
                    float2 texcoord = texcoords[index.x] * (1.0f - p.x - p.y) + texcoords[index.y] * p.x + texcoords[index.z] * p.y;
                    emission = make_float3(tex2D<float4>(texture, texcoord.x, texcoord.y));
                }

                return LightSample(pos, normalize(normal), emission * intensity, sample_pdf());
            }
            sum += areas[i];
        }
        return LightSample();
    }

    // sample proportional to area
    __device__ inline float sample_pdf()
    {
        return 1.0f / area_sum;
    }
};


class InfiniteLight
{
public:
    bool has_texture { false };
    cudaTextureObject_t texture;

    float3 emission_color;
    float intensity{ 1.0f };

public:
    __host__ __device__ InfiniteLight() {}

    __device__ float3 emission(float3 dir)
    {
        float3 color = emission_color;
        if (has_texture)
            color = make_float3(texCubemap<float4>(texture, dir.x, dir.y, dir.z));
        return color * intensity;
    }
};


// many diffuse area lights and one infinite light
class Light
{
public:
    int num;
    DiffuseAreaLight* diffuse_area_lights;
    float weight_sum;

    InfiniteLight* infinite_light;

public:
    __host__ __device__ Light() {}

    __device__ LightSample sample(float2 rnd)
    {
        float s = rnd.x * weight_sum;
        float sum = 0.0f;
        for (int i = 0; i < num; i++)
        {
            float wi = diffuse_area_lights[i].area_sum * diffuse_area_lights[i].intensity;
            if (s <= sum + wi || i == num - 1)
            {
                rnd.x = clamp((s - sum) / wi, 0.0f, 1.0f);
                LightSample ls = diffuse_area_lights[i].sample(rnd);
                ls.pdf *= wi / weight_sum;
                return ls;
            }
            sum += wi;
        }

        return LightSample();
    }

    // sample proportional to area * intensity
    __device__ float sample_pdf(int idx)
    {
        return diffuse_area_lights[idx].intensity / weight_sum;
    }

    __device__ float3 environment_emission(float3 dir)
    {
        return infinite_light->emission(dir);
    }
};