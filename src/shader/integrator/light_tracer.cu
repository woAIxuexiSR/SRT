#include <optix_device.h>
#include <device_launch_parameters.h>

#include "helper_optix.h"
#include "my_params.h"
#include "my_math.h"

extern "C" __constant__ LightTracerParams params;

template<class T>
__device__ inline T* getPRD()
{
    const unsigned u0 = optixGetPayload_0();
    const unsigned u1 = optixGetPayload_1();
    return reinterpret_cast<T*>(unpack_pointer(u0, u1));
}

__device__ inline void atomic_add_val(float4* pixels, int idx, float3 val)
{
    atomicAdd(&pixels[idx].x, val.x);
    atomicAdd(&pixels[idx].y, val.y);
    atomicAdd(&pixels[idx].z, val.z);
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

    RandomGenerator rng(params.seed + idx, 0);
    const Camera& camera = params.camera;
    const float3 camera_pos = camera.controller.pos;
    const Light* light = params.light;

    float aspect = (float)params.width / (float)params.height;
    // aspect = aspect * aspect;

    HitInfo info; int visible;
    uint2 u = pack_pointer(&info), v = pack_pointer(&visible);

    for (int i = 0; i < params.samples_per_pixel; i++)
    {
        LightSample ls = light->sample(rng.random_float2());
        float3 wi = normalize(ls.normal + uniform_sample_sphere(rng.random_float2()));
        Ray ray(ls.pos, wi);

        float cos_theta = dot(wi, ls.normal);
        float pw = cosine_hemisphere_pdf(cos_theta);
        float3 beta = ls.emission * cos_theta / ls.pdf;
        bool specular = true;
        // float cos_theta = dot(wi, ls.normal);
        // float pA = ls.pdf;
        // float pw = cosine_hemisphere_pdf(cos_theta);
        // float3 beta = ls.emission * cos_theta / pA / pw;
        // bool specular = true;
        for (int depth = 0; depth < params.max_depth; depth++)
        {
            optixTrace(params.traversable, ray.pos, ray.dir, 1e-3f, 1e16f, 0.0f,
                OptixVisibilityMask(255), OPTIX_RAY_FLAG_DISABLE_ANYHIT,
                RADIANCE_RAY_TYPE, RAY_TYPE_COUNT, RADIANCE_RAY_TYPE,
                u.x, u.y);

            // hit camera
            if (specular)
            {
                float3 to_camera = camera_pos - ray.pos;
                float t = length(to_camera);
                to_camera /= t;
                if ((!info.hit || t < length(info.pos - ray.pos)) && (dot(to_camera, ray.dir) > 1.0f - EPSILON))
                {
                    float2 xy = camera.get_xy(ray.dir);
                    if (xy.x >= 0.0f && xy.x < 1.0f && xy.y >= 0.0f && xy.y < 1.0f)
                    {
                        int idx = (int)(xy.x * params.width) + (int)(xy.y * params.height) * params.width;
                        // printf("%f %f %f, %f\n", beta.x, beta.y, beta.z, t);
                        // float3 L = aspect * beta / t / t / params.samples_per_pixel;
                        // float3 L = beta / params.samples_per_pixel;
                        // float3 L = make_float3(1.0f);
                        float3 L = beta / pw / params.samples_per_pixel;
                        atomic_add_val(params.pixels, idx, L);
                    }
                }
            }

            // miss
            if (!info.hit || info.mat->is_emissive())
                break;

            // float t = length(info.pos - ray.pos);
            float cos_i = dot(-ray.dir, info.normal);
            if(cos_i < 0.0f)
            {
                cos_i = -cos_i;
                info.normal = -info.normal;
            }

            // next event estimation for camera
            if (!info.mat->is_specular())
            {
                Ray shadow_ray(info.pos, normalize(camera_pos - info.pos));
                float2 xy = camera.get_xy(shadow_ray.dir);
                if (xy.x >= 0.0f && xy.x < 1.0f && xy.y >= 0.0f && xy.y < 1.0f)
                {
                    float t = length(camera_pos - info.pos);
                    optixTrace(params.traversable, shadow_ray.pos, shadow_ray.dir, 1e-3f, t - 1e-3f, 0.0f,
                        OptixVisibilityMask(255), OPTIX_RAY_FLAG_DISABLE_ANYHIT,
                        SHADOW_RAY_TYPE, RAY_TYPE_COUNT, SHADOW_RAY_TYPE,
                        v.x, v.y);

                    if (visible)
                    {
                        // float3 L = beta * info.mat->eval(-ray.dir, shadow_ray.dir, info.onb, info.color) * cos_theta / pdf;
                        // float3 L = beta * info.mat->eval(-ray.dir, shadow_ray.dir, info.onb, info.color) * cos_theta * cos_i / t / t / pA;
                        // float3 L = (float)params.width / (float)params.height * beta * info.mat->eval(-ray.dir, shadow_ray.dir, info.onb, info.color) * dot(shadow_ray.dir, info.normal) / t / t / pw * cos_theta;
                        float3 L = aspect * beta * info.mat->eval(-ray.dir, shadow_ray.dir, info.onb, info.color) * dot(shadow_ray.dir, info.normal) / t / t / pw;
                        int idx = (int)(xy.x * params.width) + (int)(xy.y * params.height) * params.width;
                        atomic_add_val(params.pixels, idx, L / params.samples_per_pixel);
                    }
                }
            }

            // sample next direction
            BxDFSample ms = info.mat->sample(-ray.dir, rng.random_float2(), info.onb, info.color);
            if (ms.pdf <= 1e-5f) break;
            // beta *= info.mat->eval(-ray.dir, ms.wi, info.onb, info.color) * cos_theta * cos_i / t / t / pA;
            // beta *= ms.f * dot(ms.wi, info.normal) / ms.pdf;
            beta *= info.mat->eval(-ray.dir, ms.wi, info.onb, info.color) * abs(dot(ms.wi, info.normal)) / pw; 

            // pA = pw * cos_i / t / t;
            pw = ms.pdf;
            // cos_theta = ms.cos_theta;
            specular = info.mat->is_specular();
            ray = Ray(info.pos, ms.wi);

            // russian roulette
            if (depth >= params.rr_depth)
            {
                float p = min(max(beta.x, max(beta.y, beta.z)), 0.95f);
                if (rng.random_float() > p) break;
                beta /= p;
            }
        }
    }
}