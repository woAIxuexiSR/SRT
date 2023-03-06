// #include <optix_device.h>
// #include <device_launch_parameters.h>
// #include "launchParams/LaunchParams.h"
// #include "helper_optix.h"
// #include "helper_math.h"

// extern "C" __constant__ LaunchParams<BDPTPath*> launchParams;

// struct BDPTCameraVertex
// {
//     float3 pos, wo, normal, color;
//     float3 beta;
//     const Material* mat;

//     float pdf;

//     __device__ inline void set(float3 _p, float3 _wo, float3 _n, float3 _c, float3 _b, const Material* _m)
//     {
//         pos = _p; wo = _wo; normal = _n; color = _c; beta = _b; mat = _m;
//     }
// };

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

//     RandomGenerator rng(launchParams.frameId * launchParams.height + iy, ix + 1);
//     Camera& camera = launchParams.camera;
//     BDPTPath& path = launchParams.extraData[rng.random_int(0, 1 << 20)];

//     float3 result = make_float3(0.0f);
//     for (int i = 0; i < launchParams.samplesPerPixel; i++)
//     {
//         const float xx = (ix + rng.random_float()) / launchParams.width;
//         const float yy = (iy + rng.random_float()) / launchParams.height;
//         Ray ray = camera.getRay(xx, yy);

//         BDPTCameraVertex cameraVertex[20];

//         HitInfo rayInfo;
//         int visibility;
//         thrust::pair<unsigned, unsigned> rInfoP = packPointer(&rayInfo), lInfoP = packPointer(&visibility);
//         float3 L = make_float3(0.0f), beta = make_float3(1.0f);
//         float pdf = 1.0f;
//         bool specularBounce = false;

//         int numCamera = 0;
//         for (; numCamera < 20; )
//         {
//             optixTrace(launchParams.traversable, ray.pos, ray.dir, 1e-3f, 1e16f, 0.0f,
//                 OptixVisibilityMask(255), OPTIX_RAY_FLAG_DISABLE_ANYHIT,
//                 RADIANCE_RAY_TYPE, RAY_TYPE_COUNT, RADIANCE_RAY_TYPE,
//                 rInfoP.first, rInfoP.second
//             );

//             if (numCamera == 0 || specularBounce)
//             {
//                 if (rayInfo.isHit && rayInfo.mat->isLight())
//                     L += beta * rayInfo.mat->Emission();
//                 else if (!rayInfo.isHit)
//                     L += beta * launchParams.background;
//             }

//             if (!rayInfo.isHit || rayInfo.mat->isLight()) break;


//             if (!rayInfo.mat->isGlass() && dot(rayInfo.hitNormal, ray.dir) > 0.0f)
//                 rayInfo.hitNormal = -rayInfo.hitNormal;

//             MaterialSample ms = rayInfo.mat->Sample(-ray.dir, rayInfo.hitNormal, rng.random_float2(), rayInfo.color);

//             if (rayInfo.mat->isGlass() && dot(rayInfo.hitNormal, ray.dir) > 0.0f)
//                 rayInfo.hitNormal = -rayInfo.hitNormal;

//             cameraVertex[numCamera].set(rayInfo.hitPos, -ray.dir, rayInfo.hitNormal, rayInfo.color, beta, rayInfo.mat);
//             cameraVertex[numCamera].pdf = pdf;
//             numCamera++;

//             if (ms.pdf <= 1e-5f) break;
//             beta *= ms.f * dot(ms.wi, rayInfo.hitNormal) / ms.pdf;
//             specularBounce = rayInfo.mat->isSpecular();
//             ray = Ray(rayInfo.hitPos, ms.wi);
//             pdf *= ms.pdf;

//             if (numCamera > 3)
//             {
//                 double q = fmax(beta.x, fmax(beta.y, beta.z));
//                 if (rng.random_float() > q) break;
//                 beta /= q;
//             }
//         }

//         for (int i = 0; i < numCamera; i++)
//         {
//             if (cameraVertex[i].mat->isSpecular())
//                 continue;

//             for (int j = 0; j < path.length; j++)
//             {
//                 BDPTLightVertex& vertex = path.vertices[j];

//                 float3 connectDir = vertex.pos - cameraVertex[i].pos;
//                 float dist2 = dot(connectDir, connectDir);
//                 connectDir = normalize(connectDir);

//                 if (dot(connectDir, cameraVertex[i].normal) <= 0.0f || dot(-connectDir, vertex.normal) <= 0.0f)
//                     break;

//                 optixTrace(launchParams.traversable, cameraVertex[i].pos, connectDir, 1e-3f, sqrtf(dist2) - 1e-3f, 0.0f,
//                     OptixVisibilityMask(255), OPTIX_RAY_FLAG_DISABLE_ANYHIT,
//                     SHADOW_RAY_TYPE, RAY_TYPE_COUNT, SHADOW_RAY_TYPE,
//                     lInfoP.first, lInfoP.second
//                 );
//                 if (!visibility) continue;

//                 float3 brdf = cameraVertex[i].mat->Eval(connectDir, cameraVertex[i].wo, cameraVertex[i].normal, cameraVertex[i].color);
//                 if (j != 0) brdf *= vertex.mat->Eval(vertex.wi, -connectDir, vertex.normal, vertex.color);
//                 int w = i + j + 1;
//                 if(w <= MAX_DEPTH)
//                 // int w = 1;
//                     L += cameraVertex[i].beta * brdf * vertex.beta * dot(connectDir, cameraVertex[i].normal) * dot(-connectDir, vertex.normal) / dist2 / vertex.pA / w;
//                 // L += cameraVertex[i].beta * brdf * vertex.beta * dot(connectDir, cameraVertex[i].normal) / dist2 / vertex.pA / w;
//             }
//         }

//         result += L;

//     }
//     result /= launchParams.samplesPerPixel;

//     int idx = ix + iy * launchParams.width;
//     launchParams.colorBuffer[idx] = make_float4(result, 1.0f);
// }