#include "scene/material.h"
#include "definition.h"
#include "renderer.h"

__global__ void kernel(int N)
{
    Material m;
    m.type = MaterialType::Disney;
    float3 color = make_float3(0.4, 0.5, 0.9);
    m.color = color;
    RandomGenerator rng(0, 255);
    for(int i = 1; i < 11; i++)
        m.params[i] = rng.random_float();

    float3 n = normalize(rng.random_float3());
    Onb onb(n);

    float3 wo = normalize(rng.random_float3());
    if (dot(wo, n) < 0)
        wo = -wo;

    float3 result1 = make_float3(0);
    for (int i = 0; i < N; i++)
    {
        MaterialSample ms = m.sample(wo, n, rng.random_float2(), color, false);
        if(ms.pdf <= 1e-5f) continue;
        result1 += ms.f * abs(dot(ms.wi, n)) / ms.pdf;
    }
    result1 /= N;
    printf("result1: %f %f %f\n", result1.x, result1.y, result1.z);


    float3 result2 = make_float3(0);
    for (int i = 0; i < N; i++)
    {
        // float3 wi = uniform_sample_hemisphere(rng.random_float2());
        float3 wi = uniform_sample_sphere(rng.random_float2());
        float pdf = uniform_sphere_pdf();
        // float pdf = uniform_hemisphere_pdf();
        wi = onb.to_world(wi);

        float3 f = m.eval(wi, wo, n, color, false);

        result2 += f * abs(dot(wi, n)) / pdf;
    }
    result2 /= N;
    printf("result2: %f %f %f\n", result2.x, result2.y, result2.z);

    float3 result3 = make_float3(0);
    for (int i = 0; i < N; i++)
    {
        float3 wi = cosine_sample_hemisphere(rng.random_float2());
        float pdf = cosine_hemisphere_pdf(wi.z);
        wi = onb.to_world(wi);

        float3 f = m.eval(wi, wo, n, color, false);

        result3 += f * abs(dot(wi, n)) / pdf;
    }
    result3 /= N;
    printf("result3: %f %f %f\n", result3.x, result3.y, result3.z);
}

int main()
{
    int N = 100000;
    kernel << <1, 1 >> > (N);
    checkCudaErrors(cudaDeviceSynchronize());

    return 0;
}