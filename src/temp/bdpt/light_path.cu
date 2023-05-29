// #include <optix_device.h>
// #include <device_launch_parameters.h>
// #include "launchParams/LaunchParams.h"
// #include "helper_optix.h"
// #include "helper_math.h"

// extern "C" __constant__ LaunchParams<BDPTPath*> launchParams;

// template<typename T>
// static __forceinline__ __device__ T* getPRD()
// {
//     const uint32_t u0 = optixGetPayload_0();
//     const uint32_t u1 = optixGetPayload_1();
//     return reinterpret_cast<T*>(unpackPointer(u0, u1));
// }

// extern "C" __global__ void __closesthit__radiance()
// {
//     const HitgroupData& sbtData = *(HitgroupData*)optixGetSbtDataPointer();
//     const int primID = optixGetPrimitiveIndex();
//     const float u = optixGetTriangleBarycentrics().x;
//     const float v = optixGetTriangleBarycentrics().y;
//     const float3 rayDir = optixGetWorldRayDirection();

//     const uint3& index = sbtData.index[primID];
//     const float3& A = sbtData.vertex[index.x];
//     const float3& B = sbtData.vertex[index.y];
//     const float3& C = sbtData.vertex[index.z];

//     HitInfo& prd = *(HitInfo*)getPRD<HitInfo>();
//     prd.isHit = true;
//     prd.hitPos = A * (1 - u - v) + B * u + C * v;
//     prd.mat = &sbtData.mat;
//     prd.color = sbtData.mat.getColor();

//     float3 norm;
//     if (sbtData.normal)
//         norm = sbtData.normal[index.x] * (1 - u - v) + sbtData.normal[index.y] * u + sbtData.normal[index.z] * v;
//     else
//     {
//         norm = cross(B - A, C - A);
//         if (dot(norm, rayDir) > 0.0f)
//             norm = -norm;
//     }
//     // mantain the original normal direction
//     prd.hitNormal = normalize(norm);

//     if (sbtData.hasTexture && sbtData.texcoord)
//     {
//         float2 tc = sbtData.texcoord[index.x] * (1 - u - v) + sbtData.texcoord[index.y] * u + sbtData.texcoord[index.z] * v;
//         float4 tex = tex2D<float4>(sbtData.texture, tc.x, tc.y);
//         prd.color = make_float3(pow(tex.x, 2.2f), pow(tex.y, 2.2f), pow(tex.z, 2.2f));
//     }
// }

// extern "C" __global__ void __closesthit__shadow() {}

// extern "C" __global__ void __anyhit__radiance() {}

// extern "C" __global__ void __anyhit__shadow() {}

// extern "C" __global__ void __miss__radiance()
// {
//     HitInfo& prd = *(HitInfo*)getPRD<HitInfo>();
//     prd.isHit = false;
// }

// extern "C" __global__ void __miss__shadow() {}


// extern "C" __global__ void __raygen__()
// {
//     const int ix = optixGetLaunchIndex().x;
//     const int iy = optixGetLaunchIndex().y;
//     BDPTPath& path = launchParams.extraData[ix + iy * 1024];

//     RandomGenerator rng(launchParams.frameId * launchParams.height + iy, ix);
//     Light& light = launchParams.light;

//     LightSample ls = light.Sample(rng.random_float2());
//     float3 beta = ls.emission;
//     float pdf = ls.pdf;
//     path.vertices[0].set(ls.pos, make_float3(0.0f), ls.normal, beta, beta, nullptr);
//     path.vertices[0].pA = pdf;

//     Ray ray(ls.pos, normalize(ls.normal + UniformSampleSphere(rng.random_float2())));
//     float cosTheta = dot(ray.dir, ls.normal);
//     float pw = CosineHemiSpherePdf(cosTheta);

//     int numLight = 1;
//     HitInfo hitInfo;
//     thrust::pair<unsigned, unsigned> rInfoP = packPointer(&hitInfo);
//     for(; numLight < BDPT_MAX_LIGHT_VERTICES; )
//     {
//         optixTrace(launchParams.traversable, ray.pos, ray.dir, 1e-3f, 1e16f, 0.0f,
//             OptixVisibilityMask(255), OPTIX_RAY_FLAG_DISABLE_ANYHIT,
//             RADIANCE_RAY_TYPE, RAY_TYPE_COUNT, RADIANCE_RAY_TYPE, 
//             rInfoP.first, rInfoP.second);

//         if(!hitInfo.isHit || hitInfo.mat->isLight())
//             break;
        
//         if(!hitInfo.mat->isGlass() && dot(hitInfo.hitNormal, ray.dir) > 0.0f)
//             hitInfo.hitNormal = -hitInfo.hitNormal;
        
//         MaterialSample ms = hitInfo.mat->Sample(-ray.dir, hitInfo.hitNormal, rng.random_float2(), hitInfo.color);

//         if(hitInfo.mat->isGlass() && dot(hitInfo.hitNormal, ray.dir) > 0.0f)
//             hitInfo.hitNormal = -hitInfo.hitNormal;

//         float cosThetai = dot(hitInfo.hitNormal, -ray.dir);
//         float dist = length(hitInfo.hitPos - ray.pos);
//         beta *= cosTheta * cosThetai / (dist * dist);
//         if(numLight > 1)
//             beta *= path.vertices[numLight - 1].mat->Eval(path.vertices[numLight - 1].wi, ray.dir, path.vertices[numLight - 1].normal, path.vertices[numLight - 1].color);
//         pdf = pw / dist / dist * cosThetai * path.vertices[numLight - 1].pA;
        
//         path.vertices[numLight].set(hitInfo.hitPos, -ray.dir, hitInfo.hitNormal, hitInfo.color, beta, hitInfo.mat);
//         path.vertices[numLight].pA = pdf;
//         numLight++;

//         if(ms.pdf < 1e-5f) break;
//         ray = Ray(hitInfo.hitPos, ms.wi);
//         cosTheta = dot(ms.wi, hitInfo.hitNormal);
//         pw = ms.pdf;

//         if(numLight > 3)
//         {
//             float3 L = beta / pdf;
//             float q = fmax(L.x, fmax(L.y, L.z));
//             if(rng.random_float() > q) break;
//             beta /= q;
//         }
//     }
    
//     path.length = numLight;
// }