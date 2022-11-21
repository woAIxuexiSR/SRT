#pragma once

#include <cuda_runtime.h>

class Ray
{
public:
    float3 pos;
    float3 dir;

    __host__ __device__ Ray() {}
    __host__ __device__ Ray(float3 _p, float3 _d) : pos(_p), dir(_d) {}
};