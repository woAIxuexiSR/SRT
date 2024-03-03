#pragma once

#include <cuda_runtime.h>
#include "helper_math.h"

#include "aabb.h"
#include "basic.h"
#include "matrix.h"
#include "onb.h"
#include "quaternion.h"
#include "ray.h"

/*
Transform:
1. row-major matrix
2. right-handed coordinate system
*/

class Transform
{
private:
    SquareMatrix<4> m;

public:
    __host__ __device__ Transform() {}
    __host__ __device__ Transform(const SquareMatrix<4>& _m) : m(_m) {}
    __host__ __device__ Transform(const float _m[4][4]) : Transform(SquareMatrix<4>(_m)) {}

    __host__ __device__ const SquareMatrix<4>& get_matrix() const { return m; }
    __host__ __device__ const float* operator[](int i) const { return m[i]; }
    __host__ __device__ float* operator[](int i) { return m[i]; }
    __host__ __device__ Transform operator*(const Transform& other) const { return Transform(m * other.m); }
    __host__ __device__ bool operator==(const Transform& other) const { return m == other.m; }

    /* apply transform */

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

    __host__ __device__ AABB apply_aabb(AABB box) const
    {
        AABB res;
        res.expand(apply_point(box.pmin));
        res.expand(apply_point(make_float3(box.pmin.x, box.pmin.y, box.pmax.z)));
        res.expand(apply_point(make_float3(box.pmin.x, box.pmax.y, box.pmin.z)));
        res.expand(apply_point(make_float3(box.pmin.x, box.pmax.y, box.pmax.z)));
        res.expand(apply_point(make_float3(box.pmax.x, box.pmin.y, box.pmin.z)));
        res.expand(apply_point(make_float3(box.pmax.x, box.pmin.y, box.pmax.z)));
        res.expand(apply_point(make_float3(box.pmax.x, box.pmax.y, box.pmin.z)));
        res.expand(apply_point(box.pmax));
        return res;
    }

    __host__ __device__ Onb apply_onb(const Onb& onb) const
    {
        return Onb(apply_vector(onb.z), apply_vector(onb.x), apply_vector(onb.y));
    }

    /* static methods */

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

    // anti-clockwise rotate around x-axis
    __host__ __device__ static Transform RotateX(float angle)
    {
        float sin_angle = sin(angle);
        float cos_angle = cos(angle);
        return Transform(SquareMatrix<4>(1, 0, 0, 0,
                                         0, cos_angle, -sin_angle, 0,
                                         0, sin_angle, cos_angle, 0,
                                         0, 0, 0, 1));
    }

    // anti-clockwise rotate around y-axis
    __host__ __device__ static Transform RotateY(float angle)
    {
        float sin_angle = sin(angle);
        float cos_angle = cos(angle);
        return Transform(SquareMatrix<4>(cos_angle, 0, sin_angle, 0,
                                         0, 1, 0, 0,
                                         -sin_angle, 0, cos_angle, 0,
                                         0, 0, 0, 1));
    }

    // anti-clockwise rotate around z-axis
    __host__ __device__ static Transform RotateZ(float angle)
    {
        float sin_angle = sin(angle);
        float cos_angle = cos(angle);
        return Transform(SquareMatrix<4>(cos_angle, -sin_angle, 0, 0,
                                         sin_angle, cos_angle, 0, 0,
                                         0, 0, 1, 0,
                                         0, 0, 0, 1));
    }

    // return camera to world transform
    __host__ __device__ static Transform LookAt(float3 pos, float3 target, float3 up)
    {
        float3 z = normalize(target - pos);
        float3 x = normalize(cross(up, z));
        float3 y = cross(z, x);
        return Transform(SquareMatrix<4>(x.x, y.x, z.x, pos.x,
                                         x.y, y.y, z.y, pos.y,
                                         x.z, y.z, z.z, pos.z,
                                         0, 0, 0, 1));
    }

    // quaternion to transform, quaternion must be normalized
    __host__ __device__ static Transform FromQuaternion(Quaternion quat)
    {
        float x = quat.q.x, y = quat.q.y, z = quat.q.z, w = quat.q.w;
        float x2 = x * x, y2 = y * y, z2 = z * z;
        float xy = x * y, xz = x * z, yz = y * z;
        float wx = w * x, wy = w * y, wz = w * z;

        return Transform(SquareMatrix<4>(
            1.0f - 2.0f * (y2 + z2), 2.0f * (xy - wz), 2.0f * (xz + wy), 0.0f,
            2.0f * (xy + wz), 1.0f - 2.0f * (x2 + z2), 2.0f * (yz - wx), 0.0f,
            2.0f * (xz - wy), 2.0f * (yz + wx), 1.0f - 2.0f * (x2 + y2), 0.0f,
            0.0f, 0.0f, 0.0f, 1.0f
        ));
    }

    // input: translation, rotation, scale, output: transform = T * R * S
    __host__ __device__ static Transform Compose(float3 T, Quaternion R, float3 S)
    {
        return Transform::Translate(T) * Transform::FromQuaternion(R) * Transform::Scale(S);
    }

    // input: translation, rotation, transform, output: transform = T * R * S ( S may contain shear )
    __host__ __device__ static Transform Compose(float3 T, Quaternion R, const Transform& S)
    {
        return Transform::Translate(T) * Transform::FromQuaternion(R) * S;
    }

    // input: M, output: S, R, T, where M = T * R * S ( S is not strictly a scale matrix, but a matrix with scale and shear )
    __host__ __device__ static void Decompose(const Transform& M, float3& T, Quaternion& R, Transform& S)
    {
        // extract translation
        SquareMatrix<4> Mat = M.get_matrix();
        T = make_float3(Mat[0][3], Mat[1][3], Mat[2][3]);
        Mat[0][3] = Mat[1][3] = Mat[2][3] = 0.0f;

        // polar decomposition
        float norm = 0.0f;
        int count = 0;
        SquareMatrix<4> Mk = Mat;
        do
        {
            SquareMatrix<4> Mk1 = 0.5f * (Mk + ::Inverse(Transpose(Mk)));

            norm = 0.0f;
            for (int i = 0; i < 3; i++)
            {
                float n = fabs(Mk1[i][0] - Mk[i][0]) + fabs(Mk1[i][1] - Mk[i][1]) + fabs(Mk1[i][2] - Mk[i][2]);
                norm = max(norm, n);
            }

            Mk = Mk1;
        } while (++count < 100 && norm > EPSILON);

        // extract rotation
        R = Quaternion::FromPureRotateMatrix(Mk);

        // extract scale
        S = Transform(Inverse(Mk) * Mat);
    }
};