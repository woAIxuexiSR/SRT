#pragma once

#include <cuda_runtime.h>
#include "helper_math.h"
#include "math/basic.h"
#include "math/matrix.h"
#include "math/ray.h"

class Transform
{
private:
    SquareMatrix<4> m, m_inv;

public:
    Transform() = default;

    __host__ __device__ Transform(const SquareMatrix<4>& _m) : m(_m)
    {
        thrust::optional<SquareMatrix<4> > inv = Inverse(_m);
        if(inv)
            m_inv = inv.value();
        else
            m_inv = SquareMatrix<4>();
    }

    __host__ __device__ Transform(const float _m[4][4]) : Transform(SquareMatrix<4>(_m)) {}

    __host__ __device__ Transform(const SquareMatrix<4>& _m, const SquareMatrix<4>& _m_inv) : m(_m), m_inv(_m_inv) {}

    __host__ __device__ const SquareMatrix<4>& get_matrix() const { return m; }
    __host__ __device__ const SquareMatrix<4>& get_inverse_matrix() const { return m_inv; }

    __host__ __device__ float3 apply_point(float3 p) const
    {
        float4 pi = make_float4(p, 1.0f);
        float4 po = m * pi;
        return make_float3(po) / po.w;
    }

    __host__ __device__ float3 apply_dir(float3 d) const
    {
        float4 di = make_float4(d, 0.0f);
        float4 do_ = m * di;
        return make_float3(do_);
    }

    __host__ __device__ Ray apply_ray(Ray r) const
    {
        return Ray(apply_point(r.pos), apply_dir(r.dir));
    }

    __host__ __device__ Transform operator*(const Transform& other) const
    {
        return Transform(m * other.m, other.m_inv * m_inv);
    }

};

__host__ __device__ inline Transform Inverse(const Transform& t)
{
    return Transform(t.get_inverse_matrix(), t.get_matrix());
}

__host__ __device__ inline SquareMatrix<4> Translate(float3 d)
{
    return SquareMatrix<4>(1, 0, 0, d.x,
                           0, 1, 0, d.y,
                           0, 0, 1, d.z,
                           0, 0, 0, 1);
}

__host__ __device__ inline SquareMatrix<4> Scale(float3 s)
{
    return SquareMatrix<4>(s.x, 0, 0, 0,
                           0, s.y, 0, 0,
                           0, 0, s.z, 0,
                           0, 0,   0, 1);
    
}

__host__ __device__ inline SquareMatrix<4> RotateX(float angle)
{
    float sinAngle = std::sin(angle);
    float cosAngle = std::cos(angle);
    return SquareMatrix<4>(1, 0, 0, 0,
                           0, cosAngle, -sinAngle, 0,
                           0, sinAngle, cosAngle, 0,
                           0, 0, 0, 1);
}

__host__ __device__ inline SquareMatrix<4> RotateY(float angle)
{
    float sinAngle = std::sin(angle);
    float cosAngle = std::cos(angle);
    return SquareMatrix<4>(cosAngle, 0, sinAngle, 0,
                           0, 1, 0, 0,
                           -sinAngle, 0, cosAngle, 0,
                           0, 0, 0, 1);
}

__host__ __device__ inline SquareMatrix<4> RotateZ(float angle)
{
    float sinAngle = std::sin(angle);
    float cosAngle = std::cos(angle);
    return SquareMatrix<4>(cosAngle, -sinAngle, 0, 0,
                           sinAngle, cosAngle, 0, 0,
                           0, 0, 1, 0,
                           0, 0, 0, 1);
}

__host__ __device__ inline SquareMatrix<4> LookAt(float3 pos, float3 target, float3 up)
{
    float3 z = normalize(target - pos);
    float3 x = normalize(cross(z, up));
    float3 y = cross(x, z);
    return SquareMatrix<4>(x.x, x.y, x.z, pos.x,
                           y.x, y.y, y.z, pos.y,
                           z.x, z.y, z.z, pos.z,
                           0, 0, 0, 1);
}

__host__ __device__ inline SquareMatrix<4> Othographic(float zNear, float zFar)
{
    return Scale(make_float3(1, 1, -2.0f / (zFar - zNear))) * Translate(make_float3(0, 0, -(zFar + zNear) / (zFar - zNear)));
}

__host__ __device__ inline SquareMatrix<4> Perspective(float fov, float aspect, float zNear, float zFar)
{
    SquareMatrix<4> persp(1, 0, 0, 0,
                          0, 1, 0, 0,
                          0, 0, zFar / (zFar - zNear), -zNear * zFar / (zFar - zNear),
                          0, 0, 1, 0);

    float f = 1.0f / std::tan(radians(fov) / 2.0f);
    return Scale(make_float3(f / aspect, f, 0)) * persp;
}