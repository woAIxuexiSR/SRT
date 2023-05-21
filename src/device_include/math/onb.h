#pragma once

#include "helper_math.h"

class Onb
{
private:
    float3 U, V, N;

public:
    __device__ Onb() : U({ 1, 0, 0 }), V({ 0, 1, 0 }), N({ 0, 0, 1 }) {}

    __device__ Onb(float3 _n) : N(_n)
    {
        float3 t = (abs(N.x) > abs(N.y)) ? make_float3(0, 1, 0) : make_float3(1, 0, 0);
        U = normalize(cross(t, N));
        V = cross(N, U);
    }

    __device__ float3 to_world(float3 p) const
    {
        return p.x * U + p.y * V + p.z * N;
    }

    __device__ float3 to_local(float3 p) const
    {
        return make_float3(dot(p, U), dot(p, V), dot(p, N));
    }
};