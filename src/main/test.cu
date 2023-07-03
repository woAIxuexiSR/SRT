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
    mat.base_color = make_float3(0.2f, 0.6f, 0.8f);
    mat.bxdf.type = BxDF::Type::Disney;
    mat.bxdf.metallic = 0.3f;
    mat.bxdf.sheen = 0.4f;
    mat.bxdf.clearcoat = 0.3f;
    mat.bxdf.specular = 0.5f;
    mat.bxdf.specTrans = 0.7f;


    int num = 100000;
    RandomGenerator rng(20, 0);
    float3 n = normalize(rng.random_float3());
    float3 wo = normalize(rng.random_float3());
    Onb onb(n);
    
    float3 result = make_float3(0.0f);
    for(int i = 0; i < num; i++)
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
    for(int i = 0; i < num; i++)
    {
        BxDFSample bs = mat.sample(wo, rng.random_float2(), onb, mat.base_color);
        if(bs.pdf <= EPSILON) continue;
        // result += bs.f / bs.pdf;
        result += bs.f / mat.sample_pdf(bs.wi, wo, onb, mat.base_color);
    }
    result /= num;
    printf("%f %f %f\n", result.x, result.y, result.z);

}

int main()
{
    kernel<<<1, 1>>>();
    checkCudaErrors(cudaDeviceSynchronize());

    return 0;
}