#include <optix_device.h>
#include <device_launch_parameters.h>

#include "helper_optix.h"
#include "my_params.h"
#include "my_math.h"

extern "C" __constant__ NRLHSParams params;

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

    const GInstance* instance = sbtData.instance;
    const int light_id = sbtData.light_id;

    HitInfo& prd = *(HitInfo*)getPRD<HitInfo>();
    get_hitinfo(prd, instance, prim_idx, uv, ray_dir, light_id);
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
    const int n_input_dim = 12;

    RandomGenerator rng(params.seed + idx, 0);
    const Camera& camera = params.camera;

    HitInfo info;
    uint2 u = pack_pointer(&info);

    float xx = (ix + rng.random_float()) / params.width;
    float yy = (iy + rng.random_float()) / params.height;
    Ray ray = camera.get_ray(xx, yy, rng);

    optixTrace(params.traversable, ray.pos, ray.dir, 1e-3f, 1e16f, 0.0f,
        OptixVisibilityMask(255), OPTIX_RAY_FLAG_DISABLE_ANYHIT,
        RADIANCE_RAY_TYPE, RAY_TYPE_COUNT, RADIANCE_RAY_TYPE,
        u.x, u.y);

    if (!info.hit || info.mat->is_emissive())
    {
        params.mask[idx] = 1;
        params.base[idx] = info.hit ? info.mat->emission(info.texcoord) : make_float3(0.0f);
    }
    else
    {
        params.mask[idx] = 0;
        // position
        params.inference_buffer[idx * n_input_dim + 0] = info.pos.x;
        params.inference_buffer[idx * n_input_dim + 1] = info.pos.y;
        params.inference_buffer[idx * n_input_dim + 2] = info.pos.z;
        // direction
        float3 dir = (-ray.dir + 1.f) * 0.5f;
        params.inference_buffer[idx * n_input_dim + 3] = dir.x;
        params.inference_buffer[idx * n_input_dim + 4] = dir.y;
        params.inference_buffer[idx * n_input_dim + 5] = dir.z;
        // normal
        float3 normal = (info.normal + 1.f) * 0.5f;
        params.inference_buffer[idx * n_input_dim + 6] = normal.x;
        params.inference_buffer[idx * n_input_dim + 7] = normal.y;
        params.inference_buffer[idx * n_input_dim + 8] = normal.z;
        // color
        params.inference_buffer[idx * n_input_dim + 9] = info.color.x;
        params.inference_buffer[idx * n_input_dim + 10] = info.color.y;
        params.inference_buffer[idx * n_input_dim + 11] = info.color.z;
    }
}