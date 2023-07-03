#pragma once

#include <cuda_runtime.h>

class Ray
{
public:
    float3 pos, dir;

    __host__ __device__ Ray() {}

    // build from origin and normalized direction
    __host__ __device__ Ray(float3 _p, float3 _d) : pos(_p), dir(_d) {}
};