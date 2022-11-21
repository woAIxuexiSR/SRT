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
    __device__ RandomGenerator(unsigned long long seed, unsigned long long seq)
    {
        curand_init(seed, seq, 0, &state);
    }

    __device__ float random_float() 
    {
        return 1 - curand_uniform(&state);
    }
};

// host code
__host__ inline float random_float()
{
    static std::default_random_engine g(0);
    static std::uniform_real_distribution<float> d(0, 1);
    return d(g);
}