#pragma once

#include <cuda_runtime.h>
#include "helper_math.h"
#include "math/basic.h"
#include "math/matrix.h"
#include "math/ray.h"

class Transform
{
private:
    SquareMatrix<4> m, mInv;

public:
    Transform() = default;

    __host__ __device__ Transform(const SquareMatrix<4>& _m) : m(_m)
    {
        thrust::optional<SquareMatrix<4> > inv = Inverse(_m);
        if(inv)
            mInv = inv.value();
        else
            mInv = SquareMatrix<4>();
    }

    __host__ __device__ Transform(const float _m[4][4]) : Transform(SquareMatrix<4>(_m)) {}

    __host__ __device__ Transform(const SquareMatrix<4>& _m, const SquareMatrix<4>& _mInv) : m(_m), mInv(_mInv) {}

    __host__ __device__ const SquareMatrix<4>& GetMatrix() const { return m; }
    __host__ __device__ const SquareMatrix<4>& GetInverseMatrix() const { return mInv; }

    __host__ __device__ float3 ApplyPoint(float3 p) const
    {
        float4 pi = make_float4(p, 1.0f);
        float4 po = m * pi;
        return make_float3(po) / po.w;
    }

    __host__ __device__ float3 ApplyDir(float3 d) const
    {
        float4 di = make_float4(d, 0.0f);
        float4 do_ = m * di;
        return make_float3(do_);
    }

    __host__ __device__ Ray ApplyRay(Ray r) const
    {
        return Ray(ApplyPoint(r.pos), ApplyDir(r.dir));
    }

    __host__ __device__ Transform operator*(const Transform& other) const
    {
        return Transform(m * other.m, other.mInv * mInv);
    }

};

__host__ __device__ inline Transform Inverse(const Transform& t)
{
    return Transform(t.GetInverseMatrix(), t.GetMatrix());
}

__host__ __device__ inline Transform Translate(float3 d)
{
    SquareMatrix<4> m(1, 0, 0, d.x,
                      0, 1, 0, d.y,
                      0, 0, 1, d.z,
                      0, 0, 0, 1);
    SquareMatrix<4> mInv(1, 0, 0, -d.x,
                         0, 1, 0, -d.y,
                         0, 0, 1, -d.z,
                         0, 0, 0, 1);
    return Transform(m, mInv);
}

__host__ __device__ inline Transform Scale(float3 s)
{
    SquareMatrix<4> m(s.x, 0, 0, 0,
                      0, s.y, 0, 0,
                      0, 0, s.z, 0,
                      0, 0, 0, 1);
    SquareMatrix<4> mInv(1.0f / s.x, 0, 0, 0,
                         0, 1.0f / s.y, 0, 0,
                         0, 0, 1.0f / s.z, 0,
                         0, 0, 0, 1);
    return Transform(m, mInv);
}

__host__ __device__ inline Transform RotateX(float angle)
{
    float sinAngle = std::sin(angle);
    float cosAngle = std::cos(angle);
    SquareMatrix<4> m(1, 0, 0, 0,
                      0, cosAngle, -sinAngle, 0,
                      0, sinAngle, cosAngle, 0,
                      0, 0, 0, 1);
    SquareMatrix<4> mInv(1, 0, 0, 0,
                         0, cosAngle, sinAngle, 0,
                         0, -sinAngle, cosAngle, 0,
                         0, 0, 0, 1);
    return Transform(m, mInv);
}

__host__ __device__ inline Transform RotateY(float angle)
{
    float sinAngle = std::sin(angle);
    float cosAngle = std::cos(angle);
    SquareMatrix<4> m(cosAngle, 0, sinAngle, 0,
                      0, 1, 0, 0,
                      -sinAngle, 0, cosAngle, 0,
                      0, 0, 0, 1);
    SquareMatrix<4> mInv(cosAngle, 0, -sinAngle, 0,
                         0, 1, 0, 0,
                         sinAngle, 0, cosAngle, 0,
                         0, 0, 0, 1);
    return Transform(m, mInv);
}

__host__ __device__ inline Transform RotateZ(float angle)
{
    float sinAngle = std::sin(angle);
    float cosAngle = std::cos(angle);
    SquareMatrix<4> m(cosAngle, -sinAngle, 0, 0,
                      sinAngle, cosAngle, 0, 0,
                      0, 0, 1, 0,
                      0, 0, 0, 1);
    SquareMatrix<4> mInv(cosAngle, sinAngle, 0, 0,
                         -sinAngle, cosAngle, 0, 0,
                         0, 0, 1, 0,
                         0, 0, 0, 1);
    return Transform(m, mInv);
}

__host__ __device__ inline Transform LookAt(float3 pos, float3 target, float3 up)
{
    float3 z = normalize(pos - target);
    float3 x = normalize(cross(up, z));
    float3 y = cross(z, x);
    SquareMatrix<4> m(x.x, x.y, x.z, pos.x,
                      y.x, y.y, y.z, pos.y,
                      z.x, z.y, z.z, pos.z,
                      0, 0, 0, 1);
    SquareMatrix<4> mInv(x.x, y.x, z.x, -pos.x,
                         x.y, y.y, z.y, -pos.y,
                         x.z, y.z, z.z, -pos.z,
                         0, 0, 0, 1);
    return Transform(m, mInv);
}

__host__ __device__ inline Transform Othographic(float zNear, float zFar)
{
    return Scale(make_float3(1, 1, -2.0f / (zFar - zNear))) * Translate(make_float3(0, 0, -(zFar + zNear) / (zFar - zNear)));
}

__host__ __device__ inline Transform Perspective(float fov, float aspect, float zNear, float zFar)
{
    SquareMatrix<4> persp(1, 0, 0, 0,
                          0, 1, 0, 0,
                          0, 0, zFar / (zFar - zNear), -zNear * zFar / (zFar - zNear),
                          0, 0, 1, 0);

    float f = 1.0f / std::tan(Radians(fov) / 2.0f);
    return Scale(make_float3(f / aspect, f, 0)) * Transform(persp);
}