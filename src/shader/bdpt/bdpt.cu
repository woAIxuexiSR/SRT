#include <optix_device.h>
#include <device_launch_parameters.h>
#include "launchParams/LaunchParams.h"
#include "helper_optix.h"
#include "helper_math.h"

extern "C" __constant__ LaunchParams<int> launchParams;

#define MAX_LIGHT_PATH 15

struct LightVertex
{
    float3 pos, wi, normal;
    float3 beta;
    Material mat;
    float pdf;

    __device__ void set(float3 _p, float3 _wi, float3 _n, float3 _b, Material _m, float _pdf)
    {
        pos = _p; wi = _wi; normal = _n; beta = _b; mat = _m; pdf = _pdf;
    }
};

struct HitInfo
{
    bool isHit;

    float3 hitPos;
    Material mat;
    float3 hitNormal;
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
        if(dot(norm, rayDir) > 0.0f)
            norm = -norm;
    }
    prd.hitNormal = normalize(norm);

    if (sbtData.hasTexture && sbtData.texcoord)
    {
        float2 tc = sbtData.texcoord[index.x] * (1 - u - v) + sbtData.texcoord[index.y] * u + sbtData.texcoord[index.z] * v;
        float4 tex = tex2D<float4>(sbtData.texture, tc.x, tc.y);
        float3 albedo = make_float3(pow(tex.x, 2.2f), pow(tex.y, 2.2f), pow(tex.z, 2.2f));
        prd.mat.setColor(albedo);
    }
}

extern "C" __global__ void __closesthit__shadow() 
{
    int& prd = *(int*)getPRD<int>();
    prd = 0;
}

extern "C" __global__ void __anyhit__radiance() {}

extern "C" __global__ void __anyhit__shadow() {}

extern "C" __global__ void __miss__radiance()
{
    HitInfo& prd = *(HitInfo*)getPRD<HitInfo>();
    prd.isHit = false;
}

extern "C" __global__ void __miss__shadow() 
{
    int& prd = *(int*)getPRD<int>();
    prd = 1;
}

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
        LightVertex lightPath[MAX_LIGHT_PATH];
        LightSample ls = light.Sample(rng.random_float2());

        float3 beta = ls.emission;
        float pdf = light.Pdf();
        lightPath[0].set(ls.pos, make_float3(0.0f), ls.normal, beta, Material(), pdf);

        Ray lightRay(ls.pos, normalize(ls.normal + UniformSampleSphere(rng.random_float2())));
        float cosTheta = dot(lightRay.dir, ls.normal);
        float pw = CosineHemiSpherePdf(cosTheta);

        int numLight = 1;
        HitInfo lightRayInfo;
        thrust::pair<unsigned, unsigned> lInfoP = packPointer(&lightRayInfo);
        for(int i = 0; i < MAX_LIGHT_PATH; i++)
        {
            optixTrace(launchParams.traversable, lightRay.pos, lightRay.dir, 1e-3f, 1e16f, 0.0f,
                OptixVisibilityMask(255), OPTIX_RAY_FLAG_DISABLE_ANYHIT,
                RADIANCE_RAY_TYPE, RAY_TYPE_COUNT, RADIANCE_RAY_TYPE,
                lInfoP.first, lInfoP.second
            );

            if(!lightRayInfo.isHit || lightRayInfo.mat.isLight())
                break;

            MaterialSample ms = lightRayInfo.mat.Sample(-lightRay.dir, lightRayInfo.hitNormal, rng.random_float2());
            
            if(dot(lightRayInfo.hitNormal, lightRay.dir) > 0.0f)
                lightRayInfo.hitNormal = -lightRayInfo.hitNormal;

            float cosThetai = dot(lightRayInfo.hitNormal, -lightRay.dir);
            float dist = length(lightRayInfo.hitPos - lightRay.pos);
            beta *= cosTheta * cosThetai / (dist * dist);
            if (numLight > 1)
                beta *= lightPath[numLight - 1].mat.Eval(lightPath[numLight - 1].wi, lightRay.dir, lightPath[numLight - 1].normal);
            pdf = pw / dist / dist * cosThetai * lightPath[numLight - 1].pdf;

            if(!lightRayInfo.mat.isSpecular())
            {
                lightPath[numLight].set(lightRayInfo.hitPos, -lightRay.dir, lightRayInfo.hitNormal, beta, lightRayInfo.mat, pdf);
                numLight++;
            }
            
            if(ms.pdf <= 1e-5f) break;
            lightRay = Ray(lightRayInfo.hitPos, ms.wi);
            cosTheta = dot(ms.wi, lightRayInfo.hitNormal);
            pw = ms.pdf;

            if(i > 3)
            {
                float3 L = beta / pdf;
                float q = fmax(L.x, fmax(L.y, L.z));
                if (rng.random_float() > q) break;
                beta /= q;
            }
        }

        const float xx = (ix + rng.random_float()) / launchParams.width;
        const float yy = (iy + rng.random_float()) / launchParams.height;

        HitInfo rayInfo;
        thrust::pair<unsigned, unsigned> rInfoP = packPointer(&rayInfo);
        Ray ray = camera.getRay(xx, yy);
        float3 L = make_float3(0.0f);
        beta = make_float3(1.0f);
        bool specularBounce = false;

        for(int depth = 0; depth < launchParams.MAX_DEPTH; depth++)
        {
            optixTrace(launchParams.traversable, ray.pos, ray.dir, 1e-3f, 1e16f, 0.0f,
                OptixVisibilityMask(255), OPTIX_RAY_FLAG_DISABLE_ANYHIT,
                RADIANCE_RAY_TYPE, RAY_TYPE_COUNT, RADIANCE_RAY_TYPE,
                rInfoP.first, rInfoP.second
            );

            if (depth == 0 || specularBounce)
            {
                if (rayInfo.isHit && rayInfo.mat.isLight())
                    L += beta * rayInfo.mat.Emission();
                else if (!rayInfo.isHit)
                    L += beta * launchParams.background;
            }

            if (!rayInfo.isHit || rayInfo.mat.isLight()) break;

            
            if(!rayInfo.mat.isGlass() && dot(rayInfo.hitNormal, ray.dir) > 0.0f)
                rayInfo.hitNormal = -rayInfo.hitNormal;

            if(!rayInfo.mat.isSpecular())
            {
                // for(int j = 0; j < numLight; j++)
                for(int j = 0; j < 1; j++)
                {
                    float3 connectDir = lightPath[j].pos - rayInfo.hitPos;
                    float dist2 = dot(connectDir, connectDir);
                    connectDir = normalize(connectDir);

                    if(dot(connectDir, rayInfo.hitNormal) <= 0.0f || dot(-connectDir, lightPath[j].normal) <= 0.0f)
                        continue;
                    
                    int visibility = 0;
                    thrust::pair<unsigned, unsigned> vInfoP = packPointer(&visibility);
                    optixTrace(launchParams.traversable, rayInfo.hitPos, connectDir, 1e-3f, sqrtf(dist2) - 1e-3f, 0.0f,
                        OptixVisibilityMask(255), OPTIX_RAY_FLAG_DISABLE_ANYHIT,
                        SHADOW_RAY_TYPE, RAY_TYPE_COUNT, SHADOW_RAY_TYPE,
                        vInfoP.first, vInfoP.second
                    );
                    if(!visibility) continue;
                    
                    float3 brdf = rayInfo.mat.Eval(connectDir, -ray.dir, rayInfo.hitNormal);
                    if(j != 0) brdf *= lightPath[j].mat.Eval(lightPath[j].wi, -connectDir, lightPath[j].normal);
                    // L += beta * brdf * lightPath[j].beta * dot(connectDir, rayInfo.hitNormal) * dot(-connectDir, lightPath[j].normal) / dist2 / lightPath[j].pdf / (depth + j + 1);
                    L += beta * brdf * lightPath[j].beta * dot(connectDir, rayInfo.hitNormal) * dot(-connectDir, lightPath[j].normal) / dist2 / lightPath[j].pdf;
                }
            }

            MaterialSample ms = rayInfo.mat.Sample(-ray.dir, rayInfo.hitNormal, rng.random_float2());
            
            if(rayInfo.mat.isGlass() && dot(rayInfo.hitNormal, ray.dir) > 0.0f)
                rayInfo.hitNormal = -rayInfo.hitNormal;

            if (ms.pdf <= 1e-5f) break;
            beta *= ms.f * dot(ms.wi, rayInfo.hitNormal) / ms.pdf;
            specularBounce = rayInfo.mat.isSpecular();
            ray = Ray(rayInfo.hitPos, ms.wi);

            if(depth > 3)
            {
                double q = fmax(beta.x, fmax(beta.y, beta.z));
                if(rng.random_float() > q) break;
                beta /= q;
            }
        }
        
        // for(int j = 1; j <= M; j++)
        // {
        //     if(w[j] <= 0.0f) continue;
        //     // L += contribution[j] / w[j];
        //     L += contribution[j];
        // }
        // result += clamp(L, 0.0f, 1.0f);
        result += L;

    }
    result /= launchParams.SPP;

    int idx = ix + iy * launchParams.width;
    launchParams.colorBuffer[idx] = make_float4(result, 1.0f);
}