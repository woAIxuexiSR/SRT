#include <optix_device.h>
#include <device_launch_parameters.h>

#include "helper_optix.h"
#include "my_params.h"
#include "my_math.h"

extern "C" __constant__ NRRHSParams params;

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

extern "C" __global__ void __closesthit__shadow()
{
    int& prd = *getPRD<int>();
    prd = 0;
}

extern "C" __global__ void __miss__radiance()
{
    HitInfo& prd = *(HitInfo*)getPRD<HitInfo>();
    prd.hit = false;
}

extern "C" __global__ void __miss__shadow()
{
    int& prd = *getPRD<int>();
    prd = 1;
}

extern "C" __global__ void __raygen__()
{
    const uint3 launch_idx = optixGetLaunchIndex();
    const int idx = launch_idx.x;
    const int ix = idx % params.width, iy = idx / params.width;
    const int n_input_dim = 12;

    params.mask[idx] = 1;
    params.base[idx] = make_float3(0.0f);

    RandomGenerator rng(params.seed + idx, 0);
    const Camera& camera = params.camera;
    const Light* light = params.light;

    HitInfo info; int visible;
    uint2 u = pack_pointer(&info), v = pack_pointer(&visible);

    float xx = (ix + rng.random_float()) / params.width;
    float yy = (iy + rng.random_float()) / params.height;
    Ray ray = camera.get_ray(xx, yy, rng);

    optixTrace(params.traversable, ray.pos, ray.dir, 1e-3f, 1e16f, 0.0f,
        OptixVisibilityMask(255), OPTIX_RAY_FLAG_DISABLE_ANYHIT,
        RADIANCE_RAY_TYPE, RAY_TYPE_COUNT, RADIANCE_RAY_TYPE,
        u.x, u.y);

    if (!info.hit || info.mat->is_emissive())
    {
        params.base[idx] = info.hit ? info.mat->emission(info.texcoord) : make_float3(0.0f);
        return;
    }

    // next event estimation
    if (!info.mat->is_specular())
    {
        LightSample ls = light->sample(rng.random_float2());
        Ray shadow_ray(info.pos, normalize(ls.pos - info.pos));

        float cos_o = dot(info.normal, shadow_ray.dir);
        float cos_light = dot(ls.normal, -shadow_ray.dir);
        float t = length(ls.pos - info.pos);
        if ((cos_light > 1e-4f) && (ls.pdf > 1e-4f) && (t > 1e-3f)
            && (cos_o > 0.0f || info.mat->is_transmissive()))
        {
            optixTrace(params.traversable, shadow_ray.pos, shadow_ray.dir, 1e-3f, t - 1e-3f, 0.0f,
                OptixVisibilityMask(255), OPTIX_RAY_FLAG_DISABLE_ANYHIT,
                SHADOW_RAY_TYPE, RAY_TYPE_COUNT, SHADOW_RAY_TYPE,
                v.x, v.y);

            if (visible)
            {
                float light_pdf = ls.pdf * t * t / cos_light;
                float mat_pdf = info.mat->sample_pdf(shadow_ray.dir, -ray.dir, info.onb, info.color);
                float mis_weight = light_pdf / (light_pdf + mat_pdf);
                params.base[idx] = info.mat->eval(shadow_ray.dir, -ray.dir, info.onb, info.color)
                    * abs(cos_o) * ls.emission * mis_weight / light_pdf;
            }
        }
    }

    // sample next direciton
    BxDFSample ms = info.mat->sample(-ray.dir, rng.random_float2(), info.onb, info.color);
    float cos_theta = dot(ms.wi, info.normal);
    if ((ms.pdf <= 1e-4) || (cos_theta < 0.0f && !info.mat->is_transmissive()))
        return;
    float3 weight = ms.f / ms.pdf * (ms.delta ? 1.0f : abs(cos_theta));
    ray = Ray(info.pos, ms.wi);

    optixTrace(params.traversable, ray.pos, ray.dir, 1e-3f, 1e16f, 0.0f,
        OptixVisibilityMask(255), OPTIX_RAY_FLAG_DISABLE_ANYHIT,
        RADIANCE_RAY_TYPE, RAY_TYPE_COUNT, RADIANCE_RAY_TYPE,
        u.x, u.y);

    if (!info.hit || info.mat->is_emissive())
    {
        float cos_i = dot(info.normal, -ray.dir);
        if (!info.hit || cos_i <= 1e-4f)
            return;
        float t2 = dot(info.pos - ray.pos, info.pos - ray.pos);
        float light_pdf = light->sample_pdf(info.light_id) * t2 / cos_i;
        float mis_weight = ms.pdf / (ms.pdf + light_pdf);
        params.base[idx] += weight * info.mat->emission(info.texcoord) * mis_weight;
        return;
    }
    else
    {
        params.mask[idx] = 0;
        params.weight[idx] = weight;
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