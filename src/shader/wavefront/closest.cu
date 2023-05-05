// #include <optix_device.h>
// #include <device_launch_parameters.h>

// #include "launch_params/wavefront_params.h"
// #include "helper_optix.h"
// #include "my_math.h"

// extern "C" __constant__ ClosestLaunchParams params;

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
//     prd.inner = false;
//     if (dot(prd.normal, ray_dir) > 0.0f)
//     {
//         prd.normal = -prd.normal;
//         prd.inner = true;
//     }

//     if (sbtData.has_texture && sbtData.texcoord)
//     {
//         float2 tc = sbtData.texcoord[index.x] * (1.0f - uv.x - uv.y) + sbtData.texcoord[index.y] * uv.x + sbtData.texcoord[index.z] * uv.y;
//         float4 tex = tex2D<float4>(sbtData.texture, tc.x, tc.y);
//         prd.color = make_float3(tex.x, tex.y, tex.z);
//     }
// }

// extern "C" __global__ void __closesthit__shadow() {}

// extern "C" __global__ void __anyhit__radiance() {}

// extern "C" __global__ void __anyhit__shadow() {}

// extern "C" __global__ void __miss__radiance()
// {
//     HitInfo& prd = *(HitInfo*)getPRD<HitInfo>();
//     prd.hit = false;
// }

// extern "C" __global__ void __miss__shadow() {}

// extern "C" __global__ void __raygen__()
// {
//     const uint3 launch_idx = optixGetLaunchIndex();
//     int idx = launch_idx.x;
//     if (idx >= params.queue->m_size)
//         return;

//     Ray& ray = params.ray[idx];
//     thrust::pair<unsigned, unsigned> u = pack_pointer(&params.info[idx]);
//     optixTrace(params.traversable, ray.pos, ray.dir, 1e-3f, 1e16f, 0.0f,
//         OptixVisibilityMask(255), OPTIX_RAY_FLAG_DISABLE_ANYHIT,
//         RADIANCE_RAY_TYPE, RAY_TYPE_COUNT, RADIANCE_RAY_TYPE,
//         u.first, u.second);
// }