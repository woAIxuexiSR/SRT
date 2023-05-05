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

    __device__ __host__ LightSample() {}
    __device__ __host__ LightSample(float3 _p, float3 _n, float3 _e, float _pdf) : pos(_p), normal(_n), emission(_e), pdf(_pdf) {}
};

class Light
{
public:
    int num;
    float3* vertices;
    uint3* indices;
    float3* normals;
    float* areas;
    float3* emissions;
    float total_area;

    __device__ __host__ Light() : num(0), vertices(nullptr), normals(nullptr), indices(nullptr), areas(nullptr), emissions(nullptr) {}

    __device__ __host__ void set(int _n, float3* _v, uint3* _i, float3* _vn, float* _a, float3* _e, float _ta)
    {
        num = _n;
        vertices = _v;
        indices = _i;
        normals = _vn;
        areas = _a;
        emissions = _e;
        total_area = _ta;
    }

    __device__ LightSample sample(float2 rnd)
    {
        float s = rnd.x * total_area;
        float sum = 0.0f;
        for (int i = 0; i < num; i++)
        {
            if(s <= sum + areas[i] || i == num - 1)
            {
                rnd.x = clamp((s - sum) / areas[i], 0.0f, 1.0f);
                float2 p = uniform_sample_triangle(rnd);
                uint3& index = indices[i];
                float3 pos = vertices[index.x] * (1.0f - p.x - p.y) + vertices[index.y] * p.x + vertices[index.z] * p.y;
                float3 normal = normals[index.x] * (1.0f - p.x - p.y) + normals[index.y] * p.x + normals[index.z] * p.y;
                if(length(normals[index.x]) < 0.1f)
                    normal = cross(vertices[index.y] - vertices[index.x], vertices[index.z] - vertices[index.x]);
                return LightSample(pos, normalize(normal), emissions[i], sample_pdf());
            }
        }
        return LightSample();
    }

    __device__ inline float sample_pdf()
    {
        return 1.0f / total_area;
    }
};