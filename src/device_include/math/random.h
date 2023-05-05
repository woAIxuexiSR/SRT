#pragma once

#include <random>
#include <curand.h>
#include <curand_kernel.h>

// device code
class RandomGenerator
{
private:
    curandState state;

public:
    __device__ RandomGenerator() {}

    __device__ RandomGenerator(unsigned long long seed, unsigned long long seq)
    {
        curand_init(seed, seq, 0, &state);
    }

    __device__ void init(unsigned long long seed, unsigned long long seq)
    {
        curand_init(seed, seq, 0, &state);
    }

    // [0, 1)
    __device__ float random_float()
    {
        return 1 - curand_uniform(&state);
    }

    __device__ float2 random_float2()
    {
        return make_float2(random_float(), random_float());
    }

    __device__ float3 random_float3()
    {
        return make_float3(random_float(), random_float(), random_float());
    }

    // [min, max)
    __device__ int random_int(int min, int max)
    {
        return min + curand(&state) % (max - min);
    }
};

// host code
__host__ inline float random_float()
{
    static std::default_random_engine g(0);
    static std::uniform_real_distribution<float> d(0, 1);
    return d(g);
}

__host__ inline int random_int(int min, int max)
{
    return min + random_float() * (max - min);
}