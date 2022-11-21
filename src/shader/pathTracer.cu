#include <optix_device.h>
#include <device_launch_parameters.h>
#include "launchParams/LaunchParams.h"
#include "helper_optix.h"
#include "helper_math.h"

extern "C" __constant__ LaunchParams<int> launchParams;

#define MAX_DEPTH 20
#define SPP 500000

struct PRD
{
    bool isHit;
    bool isLight;

    float3 emittance;
    DiffuseMaterial mat;
    float3 hitPos;
    float3 hitNormal;
};

template<typename T>
static __forceinline__ __device__ T* getPRD()
{
    const uint32_t u0 = optixGetPayload_0();
    const uint32_t u1 = optixGetPayload_1();
    return reinterpret_cast<T*>(unpackPointer(u0, u1));
}


extern "C" __global__ void __closesthit__radiance() {

    const HitgroupData& sbtData = *(HitgroupData*)optixGetSbtDataPointer();
    const int primID = optixGetPrimitiveIndex();
    const uint3 index = sbtData.index[primID];
    const float u = optixGetTriangleBarycentrics().x;
    const float v = optixGetTriangleBarycentrics().y;

    const float3 A = sbtData.vertex[index.x];
    const float3 B = sbtData.vertex[index.y];
    const float3 C = sbtData.vertex[index.z];

    PRD& prd = *(PRD*)getPRD<PRD>();
    prd.isHit = true;

    if (sbtData.isLight)
    {
        prd.isLight = true;
        prd.emittance = sbtData.emittance;
        return;
    }

    prd.isLight = false;
    prd.mat = sbtData.mat;
    prd.hitPos = A + u * (B - A) + v * (C - A);
    prd.hitNormal = normalize(cross(B - A, C - A));
    if (dot(prd.hitNormal, optixGetWorldRayDirection()) > 0)
        prd.hitNormal = -prd.hitNormal;

}

extern "C" __global__ void __closesthit__shadow() {}

extern "C" __global__ void __anyhit__radiance() {}

extern "C" __global__ void __anyhit__shadow() {}

extern "C" __global__ void __miss__radiance() {}

extern "C" __global__ void __miss__shadow() {}



extern "C" __global__ void __raygen__()
{
    const int ix = optixGetLaunchIndex().x;
    const int iy = optixGetLaunchIndex().y;

    RandomGenerator rng(launchParams.frameId * launchParams.height + iy, ix);

    float3 white = make_float3(1.0f, 1.0f, 1.0f);
    float3 blue = make_float3(0.5f, 0.7f, 1.0f);


    float3 result = make_float3(0.0f);
    for (int _ = 0; _ < SPP; _++)
    {
        float xx = (ix + rng.random_float()) / launchParams.width;
        float yy = (iy + rng.random_float()) / launchParams.height;
        Ray ray = launchParams.camera.getRay(xx, yy);

        float3 color = make_float3(1.0f);
        for (int depth = 0; depth < MAX_DEPTH; depth++)
        {
            PRD prd;
            prd.isHit = false;
            auto p = packPointer(&prd);

            optixTrace(
                launchParams.traversable,
                ray.pos,
                ray.dir,
                1e-3f,
                1e16f,
                0.0f,
                OptixVisibilityMask(255),
                OPTIX_RAY_FLAG_DISABLE_ANYHIT,
                RADIANCE_RAY_TYPE,
                RAY_TYPE_COUNT,
                RADIANCE_RAY_TYPE,
                p.first, p.second
            );

            if (!prd.isHit)
            {
                // float w = 0.5f * (ray.dir.y + 1.0f);
                // color *= (1.0f - w) * white + w * blue;
                color *= 0.0f;
                break;
            }

            if (prd.isLight)
            {
                color *= prd.emittance;
                break;
            }

            float2 randnum = make_float2(rng.random_float(), rng.random_float());
            MaterialSample sample = prd.mat.Sample(-ray.dir, prd.hitNormal, randnum);
            if (sample.pdf == 0.0f)
            {
                color *= 0.0f;
                break;
            }
            color *= sample.f / sample.pdf;
            ray = Ray(prd.hitPos, sample.wi);
        }
        result += color;
    }

    int idx = ix + iy * launchParams.width;
    launchParams.colorBuffer[idx] = make_float4(result / SPP, 1.0f);
}