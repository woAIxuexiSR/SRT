#include <optix_device.h>
#include <device_launch_parameters.h>

#include "helper_optix.h"
#include "my_params.h"
#include "my_math.h"

extern "C" __constant__ SimpleParams params;

template<class T>
__device__ inline T* getPRD()
{
    const unsigned u0 = optixGetPayload_0();
    const unsigned u1 = optixGetPayload_1();
    return reinterpret_cast<T*>(unpack_pointer(u0, u1));
}

extern "C" __global__ void __closesthit__radiance()
{
    const HitgroupData& sbtData = *(HitgroupData*)optixGetSbtDataPointer();
    const int prim_idx = optixGetPrimitiveIndex();
    const float2 uv = optixGetTriangleBarycentrics();
    const float3 ray_dir = optixGetWorldRayDirection();

    const GTriangleMesh* mesh = sbtData.mesh;
    const Transform* transform = sbtData.transform;
    const int light_id = sbtData.light_id;

    HitInfo& prd = *(HitInfo*)getPRD<HitInfo>();
    get_hitinfo(prd, mesh, transform, prim_idx, uv, ray_dir, light_id);
}

extern "C" __global__ void __closesthit__shadow() {}

extern "C" __global__ void __miss__radiance()
{
    HitInfo& prd = *(HitInfo*)getPRD<HitInfo>();
    prd.hit = false;
}

extern "C" __global__ void __miss__shadow() {}

extern "C" __global__ void __raygen__()
{
    const uint3 launch_idx = optixGetLaunchIndex();
    const int idx = launch_idx.x;
    const int ix = idx % params.width, iy = idx / params.width;

    RandomGenerator rng(params.seed + idx, 0);
    const Camera& camera = params.camera;

    HitInfo info;
    uint2 u = pack_pointer(&info);

    float3 result = make_float3(0.0f);
    for (int i = 0; i < params.samples_per_pixel; i++)
    {
        float xx = (ix + rng.random_float()) / params.width;
        float yy = (iy + rng.random_float()) / params.height;
        Ray ray = camera.get_ray(xx, yy, rng);

        optixTrace(params.traversable, ray.pos, ray.dir, 1e-3f, 1e16f, 0.0f,
            OptixVisibilityMask(255), OPTIX_RAY_FLAG_DISABLE_ANYHIT,
            RADIANCE_RAY_TYPE, RAY_TYPE_COUNT, RADIANCE_RAY_TYPE,
            u.x, u.y);

        float3 L = make_float3(1.0f);
        if (info.hit)
        {
            switch (params.type)
            {
            case SimpleParams::Type::Depth:
            {
                float d = length(info.pos - camera.controller.pos);
                float ratio = (d - params.min_depth) / (params.max_depth - params.min_depth);
                L = make_float3(ratio * 0.8f + 0.2f);
                break;
            }
            case SimpleParams::Type::Normal:
                L = (info.normal + 1.0f) * 0.5f;
                break;
            case SimpleParams::Type::BaseColor:
                L = info.color;
                break;
            case SimpleParams::Type::Ambient:
                L = info.color * (abs(dot(info.normal, ray.dir)) * 0.5f + 0.5f);
                break;
            case SimpleParams::Type::FaceOrientation:
                L = dot(info.normal, ray.dir) > 0.0f ? make_float3(1.0f, 0.85f, 0.0f) : make_float3(0.34f, 0.73f, 0.76f);
                break;
            }
        }

        result += L / params.samples_per_pixel;
    }

    params.pixels[idx] = make_float4(result, 1.0f);
}