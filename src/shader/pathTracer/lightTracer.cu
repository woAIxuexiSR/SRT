// #include <optix_device.h>
// #include <device_launch_parameters.h>
// #include "launchParams/LaunchParams.h"
// #include "helper_optix.h"
// #include "helper_math.h"

// extern "C" __constant__ LaunchParams<int*> launchParams;


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

// extern "C" __global__ void __closesthit__shadow()
// {
//     int& prd = *(int*)getPRD<int>();
//     prd = 0;
// }

// extern "C" __global__ void __anyhit__radiance() {}

// extern "C" __global__ void __anyhit__shadow() {}

// extern "C" __global__ void __miss__radiance()
// {
//     HitInfo& prd = *(HitInfo*)getPRD<HitInfo>();
//     prd.isHit = false;
// }

// extern "C" __global__ void __miss__shadow()
// {
//     int& prd = *(int*)getPRD<int>();
//     prd = 1;
// }

// extern "C" __global__ void __raygen__()
// {
//     const int ix = optixGetLaunchIndex().x;
//     const int iy = optixGetLaunchIndex().y;

//     RandomGenerator rng(launchParams.frameId * launchParams.height + iy, ix);
//     Camera& camera = launchParams.camera;
//     Light& light = launchParams.light;

//     float3 result = make_float3(0.0f);
//     for (int i = 0; i < launchParams.samplesPerPixel; i++)
//     {
//         HitInfo rayInfo;
//         int visible = 1;
//         thrust::pair<unsigned, unsigned> rP = packPointer(&rayInfo), vP = packPointer(&visible);

//         LightSample ls = light.Sample(rng.random_float2());
//         Ray ray(ls.pos, normalize(ls.normal + UniformSampleSphere(rng.random_float2())));

//         float cos_theta = dot(ray.dir, ls.normal);
//         // float vc = cos_theta / ls.pdf / CosineHemiSpherePdf(cos_theta);
//         // float vcm = 1.0f / CosineHemiSpherePdf(cos_theta);
//         float3 beta = ls.emission * cos_theta / ls.pdf / CosineHemiSpherePdf(cos_theta);
//         bool specularBounce = false;
//         for(int depth = 0; depth < MAX_DEPTH; depth++)
//         {
//             optixTrace(launchParams.traversable, ray.pos, ray.dir, 1e-3f, 1e16f, 0.0f,
//                 OptixVisibilityMask(255), OPTIX_RAY_FLAG_DISABLE_ANYHIT,
//                 RADIANCE_RAY_TYPE, RAY_TYPE_COUNT, RADIANCE_RAY_TYPE,
//                 rP.first, rP.second
//             );

//             if(depth == 0 || specularBounce)
//             {
//                 float3 cameraRay = normalize(camera.pos - ray.pos);
//                 if(!rayInfo.isHit && dot(cameraRay, ray.dir) > 1.0f - 1e-3f)
//                 {
//                     auto xy = camera.getXY(-cameraRay);
//                     if(xy.first >= 0.0f && xy.first <= 1.0f && xy.second >= 0.0f && xy.second <= 1.0f)
//                     {
//                         int idx = (int)(xy.first * launchParams.width) + (int)(xy.second * launchParams.height) * launchParams.width;
//                         atomicAdd(&launchParams.colorBuffer[idx].x, beta.x);
//                         atomicAdd(&launchParams.colorBuffer[idx].y, beta.y);
//                         atomicAdd(&launchParams.colorBuffer[idx].z, beta.z);
//                         atomicAdd(&launchParams.colorBuffer[idx].w, 1.0f);
//                         atomicAdd(&launchParams.extraData[idx], 1);
//                     }
//                 }
//             }

//             if(!rayInfo.isHit || rayInfo.mat->isLight()) break;

//             if(!rayInfo.mat->isGlass() && dot(rayInfo.hitNormal, ray.dir) > 0.0f)
//                 rayInfo.hitNormal = -rayInfo.hitNormal;

//             if(!rayInfo.mat->isSpecular())
//             {
//                 float3 cameraDir = rayInfo.hitPos - camera.pos;
//                 float dist2 = dot(cameraDir, cameraDir);
//                 float dist = sqrt(dist2);
//                 cameraDir /= dist;

//                 auto xy = camera.getXY(cameraDir);
//                 if(xy.first >= 0.0f && xy.first <= 1.0f && xy.second >= 0.0f && xy.second <= 1.0f)
//                 {
//                     optixTrace(launchParams.traversable, rayInfo.hitPos, -cameraDir, 1e-3f, dist - 1e-3f, 0.0f,
//                         OptixVisibilityMask(255), OPTIX_RAY_FLAG_DISABLE_ANYHIT,
//                         SHADOW_RAY_TYPE, RAY_TYPE_COUNT, SHADOW_RAY_TYPE,
//                         vP.first, vP.second
//                     );

//                     if(visible)
//                     {
//                         float3 L = beta * rayInfo.mat->Eval(-ray.dir, -cameraDir, rayInfo.hitNormal, rayInfo.color) * dot(-cameraDir, rayInfo.hitNormal) / dist2 / launchParams.samplesPerPixel;
//                         int idx = (int)(xy.first * launchParams.width) + (int)(xy.second * launchParams.height) * launchParams.width;
//                         atomicAdd(&launchParams.colorBuffer[idx].x, L.x);
//                         atomicAdd(&launchParams.colorBuffer[idx].y, L.y);
//                         atomicAdd(&launchParams.colorBuffer[idx].z, L.z);
//                         atomicAdd(&launchParams.colorBuffer[idx].w, 1.0f);
//                         atomicAdd(&launchParams.extraData[idx], 1);
//                     }
//                 }
//             }

//             MaterialSample ms = rayInfo.mat->Sample(-ray.dir, rayInfo.hitNormal, rng.random_float2(), rayInfo.color, true);

//             if(rayInfo.mat->isGlass() && dot(rayInfo.hitNormal, ray.dir) > 0.0f)
//                 rayInfo.hitNormal = -rayInfo.hitNormal;
            
//             if(ms.pdf <= 1e-5f) break;
//             beta *= ms.f * dot(ms.wi, rayInfo.hitNormal) / ms.pdf;
//             specularBounce = rayInfo.mat->isSpecular();
//             ray = Ray(rayInfo.hitPos, ms.wi);

//             if(depth > 3)
//             {
//                 float q = fmax(beta.x, fmax(beta.y, beta.z));
//                 if(rng.random_float() > q) break;
//                 beta /= q;
//             }
//         }
//     }
// }