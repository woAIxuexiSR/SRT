#pragma once

#include <cuda_runtime.h>

class AABB
{
private:
    float3 pmin, pmax;

public:
    __host__ __device__ AABB() {}
    __host__ __device__ AABB(const float3& p) : pmin(p), pmax(p) {}

    __host__ __device__ float3 get_pmin() const { return pmin; }
    __host__ __device__ float3 get_pmax() const { return pmax; }
    __host__ __device__ float3 center() const { return 0.5f * (pmin + pmax); }

    __host__ __device__ void expand(float3 p)
    {
        pmin = fminf(pmin, p);
        pmax = fmaxf(pmax, p);
    }
    __host__ __device__ void expand(const AABB& aabb)
    {
        pmin = fminf(pmin, aabb.pmin);
        pmax = fmaxf(pmax, aabb.pmax);
    }
};