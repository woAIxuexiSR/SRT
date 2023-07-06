#pragma once

#include "helper_math.h"

class Onb
{
public:
    float3 z, x, y;

public:
    __host__ __device__ Onb() : z({ 0, 0, 1 }), x({ 1, 0, 0 }), y({ 0, 1, 0 }) {}

    // build from normalized normal
    __host__ __device__ Onb(float3 _n) : z(_n)
    {
        float3 t = (abs(z.x) > abs(z.y)) ? make_float3(0, 1, 0) : make_float3(1, 0, 0);
        x = normalize(cross(t, z));
        y = cross(z, x);
    }

    // build from normalized normal and tangent
    __host__ __device__ Onb(float3 _n, float3 _t) : z(_n)
    {
        x = normalize(_t - z * dot(_t, z));
        y = cross(z, x);
    }

    // build from orthonormal basis N, T, B
    __host__ __device__ Onb(float3 _n, float3 _t, float3 _b) : z(_n), x(_t), y(_b) {}

    __host__ __device__ float3 to_world(float3 p) const
    {
        return p.x * x + p.y * y + p.z * z;
    }

    __host__ __device__ float3 to_local(float3 p) const
    {
        return make_float3(dot(p, x), dot(p, y), dot(p, z));
    }
};