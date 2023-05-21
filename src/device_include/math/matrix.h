#pragma once

#include <cuda_runtime.h>

template <int N>
__host__ __device__ inline void init(float m[N][N], int i, int j) {}

template <int N, class... Args>
__host__ __device__ inline void init(float m[N][N], int i, int j, float v, Args... args)
{
    m[i][j] = v;
    if (++j == N)
    {
        ++i;
        j = 0;
    }
    init<N>(m, i, j, args...);
}

template <int N>
__host__ __device__ inline void initDiag(float m[N][N], int i) {}

template <int N, class... Args>
__host__ __device__ inline void initDiag(float m[N][N], int i, float v, Args... args)
{
    m[i][i] = v;
    initDiag<N>(m, i + 1, args...);
}

template <int N>
class SquareMatrix
{
private:
    float m[N][N];

public:
    __host__ __device__ SquareMatrix()
    {
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                m[i][j] = (i == j) ? 1 : 0;
    }

    __host__ __device__ SquareMatrix(const float mat[N][N])
    {
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                m[i][j] = mat[i][j];
    }

    __host__ __device__ SquareMatrix(float mat[N * N])
    {
        for(int i = 0; i < N; i++)
            for(int j = 0; j < N; j++)
                m[i][j] = mat[i * N + j];
    }

    template <class... Args>
    __host__ __device__ SquareMatrix(float v, Args... args)
    {
        static_assert(1 + sizeof...(Args) == N * N,
                      "SquareMatrix constructor takes N*N arguments");
        init<N>(m, 0, 0, v, args...);
    }

    __host__ __device__ static SquareMatrix Zero()
    {
        SquareMatrix res;
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                res.m[i][j] = 0;
        return res;
    }

    template <class... Args>
    __host__ __device__ static SquareMatrix Diag(float v, Args... args)
    {
        static_assert(1 + sizeof...(Args) == N,
                      "SquareMatrix::Diag takes N arguments");
        SquareMatrix res;
        initDiag<N>(res.m, 0, v, args...);
        return res;
    }

    __host__ __device__ const float* operator[](int i) const 
    {
        return m[i];
    }

    __host__ __device__ float* operator[](int i)
    {
        return m[i];
    }

    __host__ __device__ bool operator==(const SquareMatrix& other) const
    {
        for(int i = 0; i < N; i++)
            for(int j = 0; j < N; j++)
                if(m[i][j] != other.m[i][j])
                    return false;
        return true;
    }

    __host__ __device__ SquareMatrix operator+(const SquareMatrix& other) const
    {
        SquareMatrix res;
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                res.m[i][j] = m[i][j] + other.m[i][j];
        return res;
    }

    __host__ __device__ SquareMatrix operator-(const SquareMatrix& other) const
    {
        SquareMatrix res;
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                res.m[i][j] = m[i][j] - other.m[i][j];
        return res;
    }

    __host__ __device__ SquareMatrix operator*(const SquareMatrix& other) const
    {
        SquareMatrix res = SquareMatrix::Zero();
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                for (int k = 0; k < N; k++)
                    res.m[i][j] += m[i][k] * other.m[k][j];
        return res;
    }

    __host__ __device__ SquareMatrix operator*(float s) const
    {
        SquareMatrix res;
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                res.m[i][j] = m[i][j] * s;
        return res;
    }

};

template <int N>
__host__ __device__ inline SquareMatrix<N> operator*(float s, const SquareMatrix<N>& m)
{
    return m * s;
}

__host__ __device__ inline float3 operator*(const SquareMatrix<3>& m, float3 v)
{
    return make_float3(
        m[0][0] * v.x + m[0][1] * v.y + m[0][2] * v.z,
        m[1][0] * v.x + m[1][1] * v.y + m[1][2] * v.z,
        m[2][0] * v.x + m[2][1] * v.y + m[2][2] * v.z
    );
}

__host__ __device__ inline float4 operator*(const SquareMatrix<4>& m, float4 v)
{
    return make_float4(
        m[0][0] * v.x + m[0][1] * v.y + m[0][2] * v.z + m[0][3] * v.w,
        m[1][0] * v.x + m[1][1] * v.y + m[1][2] * v.z + m[1][3] * v.w,
        m[2][0] * v.x + m[2][1] * v.y + m[2][2] * v.z + m[2][3] * v.w,
        m[3][0] * v.x + m[3][1] * v.y + m[3][2] * v.z + m[3][3] * v.w
    );
}

template<int N>
__host__ __device__ inline SquareMatrix<N> Transpose(const SquareMatrix<N>& m)
{
    SquareMatrix<N> t;
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            t[i][j] = m[j][i];
    return t;
}

template<int N>
__host__ __device__ inline float Trace(const SquareMatrix<N>& m)
{
    float trace = 0;
    for (int i = 0; i < N; i++)
        trace += m[i][i];
    return trace;
}

__host__ __device__ inline float Determinant(const SquareMatrix<2>& m)
{
    return m[0][0] * m[1][1] - m[0][1] * m[1][0];
}

__host__ __device__ inline float Determinant(const SquareMatrix<3>& m)
{
    return m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
           m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
           m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
}

__host__ __device__ inline float Determinant(const SquareMatrix<4>& m)
{
    return m[0][0] * (m[1][1] * (m[2][2] * m[3][3] - m[2][3] * m[3][2]) -
                      m[1][2] * (m[2][1] * m[3][3] - m[2][3] * m[3][1]) +
                      m[1][3] * (m[2][1] * m[3][2] - m[2][2] * m[3][1])) -
           m[0][1] * (m[1][0] * (m[2][2] * m[3][3] - m[2][3] * m[3][2]) -
                      m[1][2] * (m[2][0] * m[3][3] - m[2][3] * m[3][0]) +
                      m[1][3] * (m[2][0] * m[3][2] - m[2][2] * m[3][0])) +
           m[0][2] * (m[1][0] * (m[2][1] * m[3][3] - m[2][3] * m[3][1]) -
                      m[1][1] * (m[2][0] * m[3][3] - m[2][3] * m[3][0]) +
                      m[1][3] * (m[2][0] * m[3][1] - m[2][1] * m[3][0])) -
           m[0][3] * (m[1][0] * (m[2][1] * m[3][2] - m[2][2] * m[3][1]) -
                      m[1][1] * (m[2][0] * m[3][2] - m[2][2] * m[3][0]) +
                      m[1][2] * (m[2][0] * m[3][1] - m[2][1] * m[3][0]));
}

__host__ __device__ inline SquareMatrix<2> Inverse(const SquareMatrix<2>& m)
{
    float det = m[0][0] * m[1][1] - m[0][1] * m[1][0];
    if(det == 0)
        return SquareMatrix<2>::Zero();

    float invDet = 1.0f / det;
    SquareMatrix<2> inv;
    inv[0][0] = m[1][1] * invDet;
    inv[0][1] = -m[0][1] * invDet;
    inv[1][0] = -m[1][0] * invDet;
    inv[1][1] = m[0][0] * invDet;

    return inv;
    
}

__host__ __device__ inline SquareMatrix<3> Inverse(const SquareMatrix<3>& m)
{
    float adj00 = m[1][1] * m[2][2] - m[1][2] * m[2][1];
    float adj01 = m[1][2] * m[2][0] - m[1][0] * m[2][2];
    float adj02 = m[1][0] * m[2][1] - m[1][1] * m[2][0];

    float det = m[0][0] * adj00 + m[0][1] * adj01 + m[0][2] * adj02;
    if(det == 0)
        return SquareMatrix<3>::Zero();
    
    float invDet = 1.0f / det;
    SquareMatrix<3> inv;
    inv[0][0] = adj00 * invDet;
    inv[0][1] = (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * invDet;
    inv[0][2] = (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * invDet;
    inv[1][0] = adj01 * invDet;
    inv[1][1] = (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * invDet;
    inv[1][2] = (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * invDet;
    inv[2][0] = adj02 * invDet;
    inv[2][1] = (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * invDet;
    inv[2][2] = (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * invDet;

    return inv;
}


__host__ __device__ inline SquareMatrix<4> Inverse(const SquareMatrix<4>& m)
{
    // split into 2x2
    float s0 = m[0][0] * m[1][1] - m[0][1] * m[1][0];
    float s1 = m[0][0] * m[1][2] - m[1][0] * m[0][2];
    float s2 = m[0][0] * m[1][3] - m[1][0] * m[0][3];
    float s3 = m[0][1] * m[1][2] - m[1][1] * m[0][2];
    float s4 = m[0][1] * m[1][3] - m[1][1] * m[0][3];
    float s5 = m[0][2] * m[1][3] - m[1][2] * m[0][3];

    float c0 = m[2][0] * m[3][1] - m[2][1] * m[3][0];
    float c1 = m[2][0] * m[3][2] - m[3][0] * m[2][2];
    float c2 = m[2][0] * m[3][3] - m[3][0] * m[2][3];
    float c3 = m[2][1] * m[3][2] - m[3][1] * m[2][2];
    float c4 = m[2][1] * m[3][3] - m[3][1] * m[2][3];
    float c5 = m[2][2] * m[3][3] - m[3][2] * m[2][3];

    float det = s0 * c5 - s1 * c4 + s2 * c3 + s3 * c2 - s4 * c1 + s5 * c0;
    if(det == 0)
        return SquareMatrix<4>::Zero();
    
    float invDet = 1.0f / det;
    float inv[4][4] = {
        {   ( m[1][1] * c5 - m[1][2] * c4 + m[1][3] * c3) * invDet,
            (-m[0][1] * c5 + m[0][2] * c4 - m[0][3] * c3) * invDet,
            ( m[3][1] * s5 - m[3][2] * s4 + m[3][3] * s3) * invDet,
            (-m[2][1] * s5 + m[2][2] * s4 - m[2][3] * s3) * invDet },
        {   (-m[1][0] * c5 + m[1][2] * c2 - m[1][3] * c1) * invDet,
            ( m[0][0] * c5 - m[0][2] * c2 + m[0][3] * c1) * invDet,
            (-m[3][0] * s5 + m[3][2] * s2 - m[3][3] * s1) * invDet,
            ( m[2][0] * s5 - m[2][2] * s2 + m[2][3] * s1) * invDet },
        {   ( m[1][0] * c4 - m[1][1] * c2 + m[1][3] * c0) * invDet,
            (-m[0][0] * c4 + m[0][1] * c2 - m[0][3] * c0) * invDet,
            ( m[3][0] * s4 - m[3][1] * s2 + m[3][3] * s0) * invDet,
            (-m[2][0] * s4 + m[2][1] * s2 - m[2][3] * s0) * invDet },
        {   (-m[1][0] * c3 + m[1][1] * c1 - m[1][2] * c0) * invDet,
            ( m[0][0] * c3 - m[0][1] * c1 + m[0][2] * c0) * invDet,
            (-m[3][0] * s3 + m[3][1] * s1 - m[3][2] * s0) * invDet,
            ( m[2][0] * s3 - m[2][1] * s1 + m[2][2] * s0) * invDet}
    };

    return SquareMatrix<4>(inv);
}

template class SquareMatrix<2>;
template class SquareMatrix<3>;
template class SquareMatrix<4>;
