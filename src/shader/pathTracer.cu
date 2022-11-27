#include <optix_device.h>
#include <device_launch_parameters.h>
#include "launchParams/LaunchParams.h"
#include "helper_optix.h"
#include "helper_math.h"

extern "C" __constant__ LaunchParams<int> launchParams;

struct HitInfo
{
    bool isHit;

    float3 hitPos;
    Material mat;
    float3 hitNormal;
    float3 albedo;
};

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
    prd.mat = sbtData.mat;

    float3 norm;
    if (sbtData.normal)
        norm = sbtData.normal[index.x] * (1 - u - v) + sbtData.normal[index.y] * u + sbtData.normal[index.z] * v;
    else
    {
        norm = cross(B - A, C - A);
    }
    if (dot(norm, rayDir) > 0.0f)
        norm = -norm;
    prd.hitNormal = normalize(norm);

    if (sbtData.hasTexture && sbtData.texcoord)
    {
        float2 tc = sbtData.texcoord[index.x] * (1 - u - v) + sbtData.texcoord[index.y] * u + sbtData.texcoord[index.z] * v;
        float4 tex = tex2D<float4>(sbtData.texture, tc.x, tc.y);
        prd.albedo = make_float3(tex);
    }
    else prd.albedo = sbtData.albedo;
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
    Light& light = launchParams.light;

    float3 result = make_float3(0.0f);
    for (int i = 0; i < launchParams.SPP; i++)
    {
        float xx = (ix + rng.random_float()) / launchParams.width;
        float yy = (iy + rng.random_float()) / launchParams.height;
        Ray ray = camera.getRay(xx, yy);

        HitInfo rayInfo, lightRayInfo;
        thrust::pair<unsigned, unsigned> rInfoP = packPointer(&rayInfo), lInfoP = packPointer(&lightRayInfo);
        float3 L = make_float3(0.0f), beta = make_float3(1.0f);
        bool specularBounce = false;
        for (int depth = 0; depth < launchParams.MAX_DEPTH; depth++)
        {
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
                rInfoP.first, rInfoP.second
            );

            if (depth == 0 || specularBounce)
            {
                if (rayInfo.isHit && rayInfo.mat.isLight())
                    L += beta * rayInfo.albedo;
                else if(!rayInfo.isHit) 
                    L += beta * launchParams.background;
            }

            if (!rayInfo.isHit || rayInfo.mat.isLight()) break;

            // sample light
            float3 ls = light.Sample(rng.random_float2());
            Ray lightRay = Ray(rayInfo.hitPos, normalize(ls - rayInfo.hitPos));
            optixTrace(
                launchParams.traversable,
                lightRay.pos,
                lightRay.dir,
                1e-3f,
                1e16f,
                0.0f,
                OptixVisibilityMask(255),
                OPTIX_RAY_FLAG_DISABLE_ANYHIT,
                RADIANCE_RAY_TYPE,
                RAY_TYPE_COUNT,
                RADIANCE_RAY_TYPE,
                lInfoP.first, lInfoP.second
            );
            if(lightRayInfo.isHit && length(lightRayInfo.hitPos - ls) < 1e-3f)
            {
                float cosine = dot(lightRayInfo.hitNormal, -lightRay.dir);
                float pw = light.Pdf() * dot(ls - rayInfo.hitPos, ls - rayInfo.hitPos) / cosine;
                L += beta * rayInfo.mat.Eval(lightRay.dir, -ray.dir, rayInfo.hitNormal) * rayInfo.albedo * lightRayInfo.albedo / pw;
            }

            // sample material
            MaterialSample ms = rayInfo.mat.Sample(-ray.dir, rayInfo.hitNormal, rng.random_float2());
            if(ms.pdf <= 1e-5f) break;
            beta *= ms.f * rayInfo.albedo / ms.pdf;
            specularBounce = rayInfo.mat.isSpecular();
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
    result /= launchParams.SPP;

    int idx = ix + iy * launchParams.width;
    launchParams.colorBuffer[idx] = make_float4(result, 1.0f);
}