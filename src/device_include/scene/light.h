#pragma once

#include <cuda_runtime.h>
#include "srt_math.h"

class Light
{
public:
    int num;
    float3* vertices;
    uint3* indices;
    float* area;
    float totalArea;

    __device__ __host__ Light() : num(0), vertices(nullptr), indices(nullptr), area(nullptr) {}

    __device__ __host__ void Set(int _n, float3* _v, uint3* _i, float* _a, float _ta)
    {
        num = _n;
        vertices = _v;
        indices = _i;
        area = _a;
        totalArea = _ta;
    }

    __device__ float3 Sample(float2 sample)
    {
        float sum = 0.0f;
        for (int i = 0; i < num; i++)
        {
            sum += area[i];
            if (sample.x < sum / totalArea || i == num - 1)
            {
                sample.x = clamp((sample.x * totalArea - sum + area[i]) / area[i], 0.0f, 1.0f);
                float2 p = UniformSampleTriangle(sample);
                uint3& index = indices[i];
                return vertices[index.x] * (1.0f - p.x - p.y) + vertices[index.y] * p.x + vertices[index.z] * p.y;
            }
        }
        return vertices[0];
    }

    __device__ float Pdf()
    {
        return 1.0f / totalArea;
    }
};