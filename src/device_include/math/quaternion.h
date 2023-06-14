#pragma once

#include <cuda_runtime.h>
#include "matrix.h"

class Quaternion
{
public:
    float4 q;   // imaginary part: x, y, z; real part: w

public:
    __host__ __device__ Quaternion() : q(make_float4(0, 0, 0, 1)) {}
    __host__ __device__ Quaternion(float _x, float _y, float _z, float _w) : q(make_float4(_x, _y, _z, _w)) {}
    __host__ __device__ Quaternion(float4 _q) : q(_q) {}

    /* operator */

    __host__ __device__ Quaternion operator+(const Quaternion& other) const { return Quaternion(q + other.q); }
    __host__ __device__ Quaternion operator-(const Quaternion& other) const { return Quaternion(q - other.q); }
    __host__ __device__ Quaternion operator*(float s) const { return Quaternion(q * s); }

    __host__ __device__ Quaternion operator*(const Quaternion& other) const
    {
        return Quaternion(
            q.w * other.q.x + q.x * other.q.w + q.y * other.q.z - q.z * other.q.y,
            q.w * other.q.y + q.y * other.q.w + q.z * other.q.x - q.x * other.q.z,
            q.w * other.q.z + q.z * other.q.w + q.x * other.q.y - q.y * other.q.x,
            q.w * other.q.w - q.x * other.q.x - q.y * other.q.y - q.z * other.q.z
        );
    }

    /* static method */

    __host__ __device__ static Quaternion Identity() { return Quaternion(0, 0, 0, 1); }

    __host__ __device__ static Quaternion Conjugate(const Quaternion& quat)
    {
        return Quaternion(-quat.q.x, -quat.q.y, -quat.q.z, quat.q.w);
    }

    __host__ __device__ static float Norm(const Quaternion& quat)
    {
        return length(quat.q);
    }

    __host__ __device__ static float Norm2(const Quaternion& quat)
    {
        return dot(quat.q, quat.q);
    }

    __host__ __device__ static Quaternion Inverse(const Quaternion& quat)
    {
        return Conjugate(quat) * (1 / Norm2(quat));
    }

    __host__ __device__ static Quaternion Normalize(const Quaternion& quat)
    {
        return quat * (1 / Norm(quat));
    }

    // angle in radians
    __host__ __device__ static Quaternion FromAxisAngle(const float3& axis, float angle)
    {
        float s = sin(angle / 2);
        float c = cos(angle / 2);
        return Quaternion(axis.x * s, axis.y * s, axis.z * s, c);
    }

    // M must be pure rotation matrix
    __host__ __device__ static Quaternion FromPureRotateMatrix(const SquareMatrix<4>& M)
    {
        float trace = M[0][0] + M[1][1] + M[2][2];     // 3 - 4 * (x2 + y2 + z2)
        if(trace > 0.0f)
        {
            float s = sqrt(trace + 1.0f) * 2.0f;
            float inv_s = 1.0f / s;
            float x = (M[2][1] - M[1][2]) * inv_s;
            float y = (M[0][2] - M[2][0]) * inv_s;
            float z = (M[1][0] - M[0][1]) * inv_s;
            float w = 0.25f * s;
            return Quaternion(x, y, z, w);
        }
        else
        {
            // compute largest diagonal element (x, y, z)
            const int nxt[3] = {1, 2, 0};
            float q[3];
            int i = 0;
            if(M[1][1] > M[0][0]) i = 1;
            if(M[2][2] > M[i][i]) i = 2;
            int j = nxt[i], k = nxt[j];

            float s = sqrt((M[i][i] - (M[j][j] + M[k][k])) + 1.0f) * 2.0f;
            float inv_s = 1.0f / s;
            q[i] = 0.25f * s;
            q[j] = (M[j][i] + M[i][j]) * inv_s;
            q[k] = (M[k][i] + M[i][k]) * inv_s;
            float w = (M[k][j] - M[j][k]) * inv_s;
            return Quaternion(q[0], q[1], q[2], w);
        }
    }

    // the quaternion must be normalized, t in [0, 1]
    __host__ __device__ static Quaternion Slerp(const Quaternion& q1, const Quaternion& q2, float t)
    {
        float cos_theta = dot(q1.q, q2.q);
        float theta = acos(cos_theta);
        float theta_prime = theta * t;
        Quaternion qperp = Normalize(q2 - q1 * cos_theta);
        return q1 * cos(theta_prime) + qperp * sin(theta_prime);
    }
};