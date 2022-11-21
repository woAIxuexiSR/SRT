#pragma once

#include "helper_math.h"

// orthonormal basis
class Onb
{
private:

public:
    float3 U, V, N;

    __device__ Onb() : U(make_float3(1, 0, 0)), V(make_float3(0, 1, 0)), N(make_float3(0, 0, 1)) {}
    
    __device__ Onb(float3 _u, float3 _v, float3 _n) : U(_u), V(_v), N(_n) {}
    
    __device__ Onb(float3 _n) : N(_n)
    {
        float3 t = (abs(N.x) > abs(N.y)) ? make_float3(0, 1, 0) : make_float3(1, 0, 0);
        U = normalize(cross(t, N));
        V = cross(N, U);
    }

    __device__ float3 operator()(float3 p) const
    {
        return p.x * U + p.y * V + p.z * N;
    }
};