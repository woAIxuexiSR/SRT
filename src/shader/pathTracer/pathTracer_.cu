#include <optix_device.h>
#include <device_launch_parameters.h>
#include "launchParams/LaunchParams.h"
#include "helper_optix.h"
#include "helper_math.h"

extern "C" __constant__ LaunchParams<int> launchParams;

template<typename T>
static __forceinline__ __device__ T* getPRD()
{
    const uint32_t u0 = optixGetPayload_0();
    const uint32_t u1 = optixGetPayload_1();
    return reinterpret_cast<T*>(unpackPointer(u0, u1));
}

extern "C" __global__ void __closesthit__radiance()
{
    const HitgroupData& sbtData = *(HitgroupData*)optixGetSbtDataPointer();
    const int primID = optixGetPrimitiveIndex();
    const float u = optixGetTriangleBarycentrics().x;
    const float v = optixGetTriangleBarycentrics().y;
    const float3 rayDir = optixGetWorldRayDirection();

    const uint3& index = sbtData.index[primID];
    const float3& A = sbtData.vertex[index.x];
    const float3& B = sbtData.vertex[index.y];
    const float3& C = sbtData.vertex[index.z];

    HitInfo& prd = *(HitInfo*)getPRD<HitInfo>();
    prd.isHit = true;
    prd.hitPos = A * (1 - u - v) + B * u + C * v;
    prd.mat = &sbtData.mat;
    prd.color = sbtData.mat.getColor();

    float3 norm;
    if (sbtData.normal)
        norm = sbtData.normal[index.x] * (1 - u - v) + sbtData.normal[index.y] * u + sbtData.normal[index.z] * v;
    else
    {
        norm = cross(B - A, C - A);
        if (dot(norm, rayDir) > 0.0f)
            norm = -norm;
    }
    // mantain the original normal direction
    prd.hitNormal = normalize(norm);

    if (sbtData.hasTexture && sbtData.texcoord)
    {
        float2 tc = sbtData.texcoord[index.x] * (1 - u - v) + sbtData.texcoord[index.y] * u + sbtData.texcoord[index.z] * v;
        float4 tex = tex2D<float4>(sbtData.texture, tc.x, tc.y);
        prd.color = make_float3(pow(tex.x, 2.2f), pow(tex.y, 2.2f), pow(tex.z, 2.2f));
    }
}

extern "C" __global__ void __closesthit__shadow() {}

extern "C" __global__ void __anyhit__radiance() {}

extern "C" __global__ void __anyhit__shadow() {}

extern "C" __global__ void __miss__radiance()
{
    HitInfo& prd = *(HitInfo*)getPRD<HitInfo>();
    prd.isHit = false;
}

extern "C" __global__ void __miss__shadow() {}

extern "C" __global__ void __raygen__()
{
    const int ix = optixGetLaunchIndex().x;
    const int iy = optixGetLaunchIndex().y;

    RandomGenerator rng(launchParams.frameId * launchParams.height + iy, ix);
    Camera& camera = launchParams.camera;

    float3 result = make_float3(0.0f);
    for (int i = 0; i < launchParams.samplesPerPixel; i++)
    {
        float xx = (ix + rng.random_float()) / launchParams.width;
        float yy = (iy + rng.random_float()) / launchParams.height;
        Ray ray = camera.getRay(xx, yy);

        HitInfo rayInfo;
        thrust::pair<unsigned, unsigned> rInfoP = packPointer(&rayInfo);
        float3 L = make_float3(0.0f), beta = make_float3(1.0f);
        for (int depth = 0; depth < MAX_DEPTH; depth++)
        {
            optixTrace(launchParams.traversable, ray.pos, ray.dir, 1e-3f, 1e16f, 0.0f,
                OptixVisibilityMask(255), OPTIX_RAY_FLAG_DISABLE_ANYHIT,
                RADIANCE_RAY_TYPE, RAY_TYPE_COUNT, RADIANCE_RAY_TYPE,
                rInfoP.first, rInfoP.second
            );

            if(rayInfo.isHit && rayInfo.mat->isLight())
            {
                L += beta * rayInfo.mat->Emission();
                break;
            }
            if (!rayInfo.isHit)
            {
                L += beta * launchParams.background;
                break;
            }

            // only glass material may use opposite normal
            if(!rayInfo.mat->isGlass() && dot(rayInfo.hitNormal, ray.dir) > 0.0f)
                rayInfo.hitNormal = -rayInfo.hitNormal;

            // sample material
            MaterialSample ms = rayInfo.mat->Sample(-ray.dir, rayInfo.hitNormal, rng.random_float2(), rayInfo.color);

            // change glass normal direction after sampling
            if(rayInfo.mat->isGlass() && dot(rayInfo.hitNormal, ray.dir) > 0.0f)
                rayInfo.hitNormal = -rayInfo.hitNormal;
            
            if (ms.pdf <= 1e-5f) break;
            beta *= ms.f * dot(ms.wi, rayInfo.hitNormal) / ms.pdf;
            ray = Ray(rayInfo.hitPos, ms.wi);

            // russian roulette
            if (depth > 3)
            {
                float q = fmax(beta.x, fmax(beta.y, beta.z));
                if (rng.random_float() > q) break;
                beta /= q;
            }
        }
        // result += clamp(L, 0.0f, 1.0f);
        result += L;
    }
    result /= launchParams.samplesPerPixel;

    int idx = ix + iy * launchParams.width;
    launchParams.colorBuffer[idx] = make_float4(result, 1.0f);
}