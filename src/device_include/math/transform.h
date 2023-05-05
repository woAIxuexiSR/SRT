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

    __host__ __device__ const SquareMatrix<4>& get_matrix() const { return m; }
    
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
        return Transform(m * other.m);
    }
};

__host__ __device__ inline Transform Inverse(const Transform& t)
{
    thrust::optional<SquareMatrix<4> > inv = Inverse(t.get_matrix());
    if(inv) 
        return Transform(inv.value());
    return Transform();
}

__host__ __device__ inline Transform Translate(float3 d)
{
    return Transform(SquareMatrix<4>(1, 0, 0, d.x,
                                     0, 1, 0, d.y,
                                     0, 0, 1, d.z,
                                     0, 0, 0, 1));
}

__host__ __device__ inline Transform Scale(float3 s)
{
    return Transform(SquareMatrix<4>(s.x, 0, 0, 0,
                                     0, s.y, 0, 0,
                                     0, 0, s.z, 0,
                                     0, 0,   0, 1));
}

__host__ __device__ inline Transform RotateX(float angle)
{
    float sin_angle = sin(angle);
    float cos_angle = cos(angle);
    return Transform(SquareMatrix<4>(1, 0, 0, 0,
                                     0, cos_angle, -sin_angle, 0,
                                     0, sin_angle, cos_angle, 0,
                                     0, 0, 0, 1));

}

__host__ __device__ inline Transform RotateY(float angle)
{
    float sin_angle = sin(angle);
    float cos_angle = cos(angle);
    return Transform(SquareMatrix<4>(cos_angle, 0, sin_angle, 0,
                                     0, 1, 0, 0,
                                     -sin_angle, 0, cos_angle, 0,
                                     0, 0, 0, 1));
}

__host__ __device__ inline Transform RotateZ(float angle)
{
    float sin_angle = sin(angle);
    float cos_angle = cos(angle);
    return Transform(SquareMatrix<4>(cos_angle, -sin_angle, 0, 0,
                                     sin_angle, cos_angle, 0, 0,
                                     0, 0, 1, 0,
                                     0, 0, 0, 1));
}

__host__ __device__ inline Transform LookAt(float3 pos, float3 target, float3 up)
{
    float3 z = normalize(target - pos);
    float3 x = normalize(cross(z, up));
    float3 y = cross(x, z);
    return Transform(SquareMatrix<4>(x.x, x.y, x.z, pos.x,
                                     y.x, y.y, y.z, pos.y,
                                     z.x, z.y, z.z, pos.z,
                                     0, 0, 0, 1));
}

__host__ __device__ inline Transform Orthographic(float near, float far)
{
    return Scale(make_float3(1, 1, -2.0f / (far - near))) * Translate(make_float3(0, 0, -(far + near) / (far - near)));
}

__host__ __device__ inline Transform Perspective(float fov, float aspect, float near, float far)
{
    Transform persp(SquareMatrix<4>(1, 0, 0, 0,
                                    0, 1, 0, 0,
                                    0, 0, far / (far - near), -near * far / (far - near),
                                    0, 0, 1, 0));
    float f = 1.0f / tan(radians(fov) / 2.0f);
    return Scale(make_float3(f / aspect, f, 0)) * persp;
}