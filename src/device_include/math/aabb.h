#pragma once

#include "basic.h"

class AABB
{
public:
    float3 pmin, pmax;

public:
    __host__ __device__ AABB() : pmin(make_float3(FLOAT_MAX)), pmax(make_float3(FLOAT_MIN)) {}
    __host__ __device__ AABB(const float3& p) : pmin(p), pmax(p) {}

    __host__ __device__ float3 center() const { return 0.5f * (pmin + pmax); }

    __host__ __device__ bool inside(float3 p) const
    {
        return p.x >= pmin.x && p.x <= pmax.x &&
               p.y >= pmin.y && p.y <= pmax.y &&
               p.z >= pmin.z && p.z <= pmax.z;
    }

    // calculate the max distance from p to the AABB
    __host__ __device__ float max_distance(float3 p) const
    {
        float x = fmaxf(fabs(p.x - pmin.x), fabs(p.x - pmax.x));
        float y = fmaxf(fabs(p.y - pmin.y), fabs(p.y - pmax.y));
        float z = fmaxf(fabs(p.z - pmin.z), fabs(p.z - pmax.z));
        return length(make_float3(x, y, z));
    }

    // calculate the min distance from p to the AABB
    __host__ __device__ float min_distance(float3 p) const
    {
        if (inside(p)) return 0.0f;
        float x = fminf(fabs(p.x - pmin.x), fabs(p.x - pmax.x));
        float y = fminf(fabs(p.y - pmin.y), fabs(p.y - pmax.y));
        float z = fminf(fabs(p.z - pmin.z), fabs(p.z - pmax.z));
        return length(make_float3(x, y, z));
    }

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