// #include <optix_device.h>
// #include <device_launch_parameters.h>

// #include "helper_optix.h"
// #include "my_params.h"
// #include "my_math.h"

// extern "C" __constant__ ClosestParams params;

// template<class T>
// __device__ inline T* getPRD()
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

//     const GTriangleMesh* mesh = sbtData.mesh;
//     const Transform* transform = sbtData.transform;
//     const int light_id = sbtData.light_id;

//     HitInfo& prd = *(HitInfo*)getPRD<HitInfo>();
//     get_hitinfo(prd, mesh, transform, prim_idx, uv, ray_dir, light_id);
// }

// extern "C" __global__ void __closesthit__shadow() {}

// extern "C" __global__ void __miss__radiance()
// {
//     HitInfo& prd = *(HitInfo*)getPRD<HitInfo>();
//     prd.hit = false;
// }

// extern "C" __global__ void __miss__shadow() {}

// extern "C" __global__ void __raygen__()
// {
//     const uint3 launch_idx = optixGetLaunchIndex();
//     const int idx = launch_idx.x;

//     if (idx >= params.ray_queue->size())
//         return;

//     const RayWorkItem& item = params.ray_queue->fetch(idx);

//     HitInfo info;
//     uint2 u = pack_pointer(&info);
//     optixTrace(params.traversable, item.ray.pos, item.ray.dir, 1e-3f, 1e16f, 0.0f,
//         OptixVisibilityMask(255), OPTIX_RAY_FLAG_DISABLE_ANYHIT,
//         RADIANCE_RAY_TYPE, RAY_TYPE_COUNT, RADIANCE_RAY_TYPE,
//         u.x, u.y);

//     if (!info.hit)
//     {
//         MissWorkItem miss_item;
//         miss_item.idx = item.idx;
//         miss_item.beta = item.beta;
//         miss_item.dir = item.ray.dir;
//         params.miss_queue->push(miss_item);
//     }
//     else if (info.mat->is_emissive())
//     {
//         HitLightWorkItem hit_light_item;
//         hit_light_item.idx = item.idx;
//         hit_light_item.ray = item.ray;
//         hit_light_item.beta = item.beta;
//         hit_light_item.pdf = item.pdf;
//         hit_light_item.specular = item.specular;
//         hit_light_item.pos = info.pos;
//         hit_light_item.normal = info.normal;
//         hit_light_item.texcoord = info.texcoord;
//         hit_light_item.mat = info.mat;
//         hit_light_item.light_id = info.light_id;
//         params.hit_light_queue->push(hit_light_item);
//     }
//     else
//     {
//         ScatterRayWorkItem scatter_item;
//         scatter_item.idx = item.idx;
//         scatter_item.ray = item.ray;
//         scatter_item.beta = item.beta;
//         scatter_item.pos = info.pos;
//         scatter_item.normal = info.normal;
//         scatter_item.color = info.color;
//         scatter_item.onb = info.onb;
//         scatter_item.mat = info.mat;
//         params.scatter_ray_queue->push(scatter_item);
//     }
// }