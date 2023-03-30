// #include <optix_device.h>
// #include <device_launch_parameters.h>

// #include "launch_params/launch_params.h"
// #include "helper_optix.h"
// #include "my_math.h"

// extern "C" __constant__ LaunchParams<int> params;

// template<class T>
// static __forceinline__ __device__ T* getPRD()
// {
//     const unsigned u0 = optixGetPayload_0();
//     const unsigned u1 = optixGetPayload_1();
//     return reinterpret_cast<T*>(unpack_pointer(u0, u1));
// }

// extern "C" __global__ void __closesthit__radiance()
// {
//     const HitgroupData& sbtData = *(HitgroupData*)optixGetSbtDataPointer();
//     const int prim_idx = optixGetPrimitiveIndex();
//     const float2 uv = optixGetTriangleBarycentrics();
//     const float3 ray_dir = optixGetWorldRayDirection();

//     const uint3& index = sbtData.index[prim_idx];
//     const float3& v0 = sbtData.vertex[index.x];
//     const float3& v1 = sbtData.vertex[index.y];
//     const float3& v2 = sbtData.vertex[index.z];

//     HitInfo& prd = *(HitInfo*)getPRD<HitInfo>();
//     prd.hit = true;
//     prd.pos = v0 * (1.0f - uv.x - uv.y) + v1 * uv.x + v2 * uv.y;
//     prd.mat = &sbtData.mat;
//     prd.color = sbtData.mat.get_color();

//     float3 norm;
//     if (sbtData.normal)
//         norm = sbtData.normal[index.x] * (1.0f - uv.x - uv.y) + sbtData.normal[index.y] * uv.x + sbtData.normal[index.z] * uv.y;
//     else
//         norm = cross(v1 - v0, v2 - v0);
//     prd.normal = normalize(norm);
//     if (!prd.mat->is_glass() && dot(prd.normal, ray_dir) > 0.0f)
//         prd.normal = -prd.normal;

//     if (sbtData.has_texture && sbtData.texcoord)
//     {
//         float2 tc = sbtData.texcoord[index.x] * (1.0f - uv.x - uv.y) + sbtData.texcoord[index.y] * uv.x + sbtData.texcoord[index.z] * uv.y;
//         float4 tex = tex2D<float4>(sbtData.texture, tc.x, tc.y);
//         prd.color = make_float3(tex.x, tex.y, tex.z);
//     }
// }

// extern "C" __global__ void __closesthit__shadow()
// {
//     int& prd = *getPRD<int>();
//     prd = 0;
// }

// extern "C" __global__ void __anyhit__radiance() {}

// extern "C" __global__ void __anyhit__shadow() {}

// extern "C" __global__ void __miss__radiance()
// {
//     HitInfo& prd = *(HitInfo*)getPRD<HitInfo>();
//     prd.hit = false;
// }

// extern "C" __global__ void __miss__shadow()
// {
//     int& prd = *getPRD<int>();
//     prd = 1;
// }

// __device__ inline void __atomic_add_f4(float4* buffer, int idx, float4 val)
// {
//     atomicAdd(&buffer[idx].x, val.x);
//     atomicAdd(&buffer[idx].y, val.y);
//     atomicAdd(&buffer[idx].z, val.z);
//     atomicAdd(&buffer[idx].w, val.w);
// }

// extern "C" __global__ void __raygen__()
// {
//     const uint3 launch_idx = optixGetLaunchIndex();
//     const int ix = launch_idx.x, iy = launch_idx.y;

//     RandomGenerator rng(params.frame * params.height + iy, ix);
//     Camera& camera = params.camera;
//     Light& light = params.light;

//     HitInfo info; int visible = 1;
//     thrust::pair<unsigned, unsigned> u = pack_pointer(&info), v = pack_pointer(&visible);

//     for (int i = 0; i < params.spp; i++)
//     {
//         LightSample ls = light.sample(rng.random_float2());
//         Ray ray(ls.pos, normalize(ls.normal + uniform_sample_sphere(rng.random_float2())));

//         float cos_theta = dot(ray.dir, ls.normal);
//         float3 beta = ls.emission * cos_theta / ls.pdf / cosine_hemisphere_pdf(cos_theta);
//         bool specular = true;
//         for (int depth = 0; depth < MAX_DEPTH; depth++)
//         {
//             optixTrace(params.traversable, ray.pos, ray.dir, 1e-3f, 1e16f, 0.0f,
//                 OptixVisibilityMask(255), OPTIX_RAY_FLAG_DISABLE_ANYHIT,
//                 RADIANCE_RAY_TYPE, RAY_TYPE_COUNT, RADIANCE_RAY_TYPE,
//                 u.first, u.second);

//             if (specular)
//             {
//                 float3 to_camera = camera.pos - ray.pos;
//                 if ((!info.hit || length(to_camera) < length(info.pos - ray.pos))
//                     && dot(normalize(to_camera), ray.dir) > 1.0f - 1e-3f)
//                 {
//                     auto xy = camera.get_xy(-to_camera);
//                     float x = xy.first, y = xy.second;
//                     if (x >= 0.0f && x <= 1.0f && y >= 0.0f && y <= 1.0f)
//                     {
//                         int idx = (int)(x * params.width) + (int)(y * params.height) * params.width;
//                         float4 L = make_float4(beta, 1.0f) / params.spp;
//                         __atomic_add_f4(params.buffer, idx, L);
//                     }
//                 }
//             }

//             if (!info.hit || info.mat->is_light())
//                 break;

//             if (!info.mat->is_specular())
//             {
//                 Ray shadow_ray(info.pos, normalize(camera.pos - info.pos));
//                 auto xy = camera.get_xy(-shadow_ray.dir);
//                 float x = xy.first, y = xy.second;
//                 if (x >= 0.0f && x <= 1.0f && y >= 0.0f && y <= 1.0f)
//                 {
//                     float t = length(camera.pos - info.pos);
//                     optixTrace(params.traversable, shadow_ray.pos, shadow_ray.dir, 1e-3f, t - 1e-3f, 0.0f,
//                         OptixVisibilityMask(255), OPTIX_RAY_FLAG_DISABLE_ANYHIT,
//                         SHADOW_RAY_TYPE, RAY_TYPE_COUNT, SHADOW_RAY_TYPE,
//                         v.first, v.second);

//                     if (visible)
//                     {
//                         float3 L = beta * info.mat->eval(-ray.dir, shadow_ray.dir, info.normal, info.color)
//                             * dot(shadow_ray.dir, info.normal) / t / t;
//                         int idx = (int)(x * params.width) + (int)(y * params.height) * params.width;
//                         __atomic_add_f4(params.buffer, idx, make_float4(L, 1.0f) / params.spp);
//                     }
//                 }
//             }

//             MaterialSample ms = info.mat->sample(-ray.dir, info.normal, rng.random_float2(), info.color);
//             if (ms.pdf <= 1e-6f) break;
//             beta *= ms.f * dot(ms.wi, info.normal) / ms.pdf;
//             specular = info.mat->is_specular();
//             ray = Ray(info.pos, ms.wi);

//             if (depth >= 3)
//             {
//                 float q = max(max(beta.x, beta.y), beta.z);
//                 if (rng.random_float() > q) break;
//                 beta /= q;
//             }
//         }
//     }
// }