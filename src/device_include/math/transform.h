#pragma once

#include <cuda_runtime.h>
#include "helper_math.h"
#include "math/basic.h"
#include "math/matrix.h"
#include "math/ray.h"

class Transform
{
private:
    SquareMatrix<4> m;

public:
    Transform() = default;
    __host__ __device__ Transform(const SquareMatrix<4>& _m) : m(_m) {}
    __host__ __device__ Transform(const float _m[4][4]) : Transform(SquareMatrix<4>(_m)) {}

    __host__ __device__ const float* operator[](int i) const { return m[i]; }
    __host__ __device__ float* operator[](int i) { return m[i]; }
    __host__ __device__ Transform operator*(const Transform& other) const { return Transform(m * other.m); }
    __host__ __device__ bool operator==(const Transform& other) const { return m == other.m; }

    // apply transform
    __host__ __device__ float3 apply_point(float3 p) const
    {
        return make_float3(
            m[0][0] * p.x + m[0][1] * p.y + m[0][2] * p.z + m[0][3],
            m[1][0] * p.x + m[1][1] * p.y + m[1][2] * p.z + m[1][3],
            m[2][0] * p.x + m[2][1] * p.y + m[2][2] * p.z + m[2][3]
        );
    }
    __host__ __device__ float3 apply_vector(float3 v) const
    {
        return make_float3(
            m[0][0] * v.x + m[0][1] * v.y + m[0][2] * v.z,
            m[1][0] * v.x + m[1][1] * v.y + m[1][2] * v.z,
            m[2][0] * v.x + m[2][1] * v.y + m[2][2] * v.z
        );
    }
    __host__ __device__ Ray apply_ray(Ray r) const
    {
        return Ray(apply_point(r.pos), apply_vector(r.dir));
    }

    // static methods
    __host__ __device__ static Transform Inverse(const Transform& t)
    {
        return Transform(::Inverse(t.m));
    }

    __host__ __device__ static Transform Identity() { return Transform(); }

    __host__ __device__ static Transform Translate(float3 d)
    {
        return Transform(SquareMatrix<4>(1, 0, 0, d.x,
                                         0, 1, 0, d.y,
                                         0, 0, 1, d.z,
                                         0, 0, 0, 1));
    }

    __host__ __device__ static Transform Scale(float3 s)
    {
        return Transform(SquareMatrix<4>(s.x, 0, 0, 0,
                                         0, s.y, 0, 0,
                                         0, 0, s.z, 0,
                                         0, 0, 0, 1));
    }

    __host__ __device__ static Transform RotateX(float angle)
    {
        float sin_angle = sin(angle);
        float cos_angle = cos(angle);
        return Transform(SquareMatrix<4>(1, 0, 0, 0,
                                         0, cos_angle, -sin_angle, 0,
                                         0, sin_angle, cos_angle, 0,
                                         0, 0, 0, 1));
    }

    __host__ __device__ static Transform RotateY(float angle)
    {
        float sin_angle = sin(angle);
        float cos_angle = cos(angle);
        return Transform(SquareMatrix<4>(cos_angle, 0, sin_angle, 0,
                                         0, 1, 0, 0,
                                         -sin_angle, 0, cos_angle, 0,
                                         0, 0, 0, 1));
    }

    __host__ __device__ static Transform RotateZ(float angle)
    {
        float sin_angle = sin(angle);
        float cos_angle = cos(angle);
        return Transform(SquareMatrix<4>(cos_angle, -sin_angle, 0, 0,
                                         sin_angle, cos_angle, 0, 0,
                                         0, 0, 1, 0,
                                         0, 0, 0, 1));
    }

    __host__ __device__ static Transform LookAt(float3 pos, float3 target, float3 up)
    {
        // return camera to world transform
        float3 z = normalize(target - pos);
        float3 x = normalize(cross(up, z));
        float3 y = cross(z, x);
        return Transform(SquareMatrix<4>(x.x, y.x, z.x, pos.x,
                                         x.y, y.y, z.y, pos.y,
                                         x.z, y.z, z.z, pos.z,
                                         0, 0, 0, 1));
    }
};