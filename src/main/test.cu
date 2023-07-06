#include <iostream>
#include <cuda_runtime.h>
#include "my_math.h"
#include "scene/gmesh.h"
#include "scene/light.h"
#include "scene/camera.h"
#include "scene.h"

using namespace std;

__global__ void kernel()
{
    GMaterial mat;
    mat.base_color = make_float3(0.14f, 0.45f, 0.091f);
    mat.bxdf.type = BxDF::Type::Disney;
    mat.bxdf.metallic = 0.0f;
    mat.bxdf.roughness = 0.5f;
    mat.bxdf.specular = 0.5f;


    int num = 100000;
    RandomGenerator rng(70, 0);
    float3 n = normalize(rng.random_float3());
    float3 wo = normalize(rng.random_float3());
    Onb onb(n);

    float3 result = make_float3(0.0f);
    for (int i = 0; i < num; i++)
    {
        // float3 wi = onb.to_world(uniform_sample_hemisphere(rng.random_float2()));
        // float3 t = mat.eval(wi, wo, onb, mat.base_color);
        // result += t / uniform_hemisphere_pdf();
        float3 wi = onb.to_world(uniform_sample_sphere(rng.random_float2()));
        float3 t = mat.eval(wi, wo, onb, mat.base_color);
        result += t / uniform_sphere_pdf();
    }
    result /= num;
    printf("%f %f %f\n", result.x, result.y, result.z);


    result = make_float3(0.0f);
    for (int i = 0; i < num; i++)
    {
        BxDFSample bs = mat.sample(wo, rng.random_float2(), onb, mat.base_color);
        if (bs.pdf <= EPSILON) continue;
        result += bs.f / bs.pdf;
        // result += bs.f / mat.sample_pdf(bs.wi, wo, onb, mat.base_color);
    }
    result /= num;
    printf("%f %f %f\n", result.x, result.y, result.z);

}

float f(float3 v) { return v.z; }

int main()
{
    kernel<<<1, 1>>>();
    checkCudaErrors(cudaDeviceSynchronize());

    // int n = 1000000;
    // float3 V = normalize(make_float3(0.2, 0.4, 0.6));
    // float result = 0.0f;
    // for (int i = 0; i < n; i++)
    // {
    //     float3 v = uniform_sample_hemisphere(make_float2(random_float(), random_float()));
    //     if(dot(V, v) >= 0.0f)
    //         result += f(v) / uniform_hemisphere_pdf();
    // }
    // cout << result / n << endl;

    // result = 0.0f;
    // for (int i = 0; i < n; i++)
    // {
    //     // float3 v = sample_GTR2_aniso(2.0f, 2.0f, make_float2(random_float(), random_float()));
    //     // result += f(v) / (GTR2_aniso(v.z, v.x, v.y, 2.0f, 2.0f) * v.z);
    //     // float3 v = sample_GTR2(0.3f, make_float2(random_float(), random_float()));
    //     // result += f(v) / (GTR2(v.z, 0.3f) * v.z);
    //     float3 v = sample_GGXVNDF(V, 0.3f, make_float2(random_float(), random_float()));
    //     float pdf = GTR2(v.z, 0.3f) * max(dot(V, v), 0.0f) * smithG_GGX(V.z, 0.3f) / V.z;
    //     result += f(v) / pdf;
    // }
    // cout << result / n << endl;

    return 0;
}