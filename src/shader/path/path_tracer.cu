#include <optix_device.h>
#include <device_launch_parameters.h>

#include "launch_params/launch_params.h"
#include "helper_optix.h"
#include "my_math.h"

extern "C" __constant__ LaunchParams<int> params;

template<class T>
static __forceinline__ __device__ T* getPRD()
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

    const uint3& index = sbtData.index[prim_idx];
    const float3& v0 = sbtData.vertex[index.x];
    const float3& v1 = sbtData.vertex[index.y];
    const float3& v2 = sbtData.vertex[index.z];

    HitInfo& prd = *(HitInfo*)getPRD<HitInfo>();
    prd.hit = true;
    prd.pos = v0 * (1.0f - uv.x - uv.y) + v1 * uv.x + v2 * uv.y;
    prd.mat = &sbtData.mat;
    prd.color = sbtData.mat.get_color();

    float3 norm;
    if (sbtData.normal)
        norm = sbtData.normal[index.x] * (1.0f - uv.x - uv.y) + sbtData.normal[index.y] * uv.x + sbtData.normal[index.z] * uv.y;
    else
        norm = cross(v1 - v0, v2 - v0);
    prd.normal = normalize(norm);
    if(!prd.mat->is_transmissive() && dot(prd.normal, ray_dir) > 0.0f)
        prd.normal = -prd.normal;

    if (sbtData.has_texture && sbtData.texcoord)
    {
        float2 tc = sbtData.texcoord[index.x] * (1.0f - uv.x - uv.y) + sbtData.texcoord[index.y] * uv.x + sbtData.texcoord[index.z] * uv.y;
        float4 tex = tex2D<float4>(sbtData.texture, tc.x, tc.y);
        prd.color = make_float3(tex.x, tex.y, tex.z);
    }
}

extern "C" __global__ void __closesthit__shadow()
{
    int& prd = *getPRD<int>();
    prd = 0;
}

extern "C" __global__ void __anyhit__radiance() {}

extern "C" __global__ void __anyhit__shadow() {}

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
    const int ix = launch_idx.x, iy = launch_idx.y;

    RandomGenerator rng(params.frame * params.height + iy, ix);
    Camera& camera = params.camera;
    Light& light = params.light;

    HitInfo info; int visible = 1;
    thrust::pair<unsigned, unsigned> u = pack_pointer(&info), v = pack_pointer(&visible);

    float3 result = make_float3(0.0f);
    for (int i = 0; i < params.spp; i++)
    {
        float xx = (ix + rng.random_float()) / params.width;
        float yy = (iy + rng.random_float()) / params.height;
        Ray ray = camera.get_ray(xx, yy);

        float3 L = make_float3(0.0f), beta = make_float3(1.0f);
        bool specular = true;
        for (int depth = 0; depth < MAX_DEPTH; depth++)
        {
            optixTrace(params.traversable, ray.pos, ray.dir, 1e-3f, 1e16f, 0.0f,
                OptixVisibilityMask(255), OPTIX_RAY_FLAG_DISABLE_ANYHIT,
                RADIANCE_RAY_TYPE, RAY_TYPE_COUNT, RADIANCE_RAY_TYPE,
                u.first, u.second);

            if (specular)
            {
                if (info.hit && info.mat->is_emissive())
                    L += beta * info.mat->get_emission();
                else if (!info.hit)
                    L += beta * params.background;
            }

            if (!info.hit || info.mat->is_emissive())
                break;

            if (!info.mat->is_specular())
            {
                LightSample ls = light.sample(rng.random_float2());
                Ray shadow_ray(info.pos, normalize(ls.pos - info.pos));

                float cos_theta = dot(ls.normal, -shadow_ray.dir);
                if (cos_theta > 0.0f)
                {
                    float t = length(ls.pos - info.pos);
                    optixTrace(params.traversable, shadow_ray.pos, shadow_ray.dir, 1e-3f, t - 1e-3f, 0.0f,
                        OptixVisibilityMask(255), OPTIX_RAY_FLAG_DISABLE_ANYHIT,
                        SHADOW_RAY_TYPE, RAY_TYPE_COUNT, SHADOW_RAY_TYPE,
                        v.first, v.second);

                    if (visible)
                        L += beta * info.mat->eval(shadow_ray.dir, -ray.dir, info.normal, info.color)
                            * dot(info.normal, shadow_ray.dir) * ls.emission / (ls.pdf * t * t / cos_theta);
                }
            }

            MaterialSample ms = info.mat->sample(-ray.dir, info.normal, rng.random_float2(), info.color);
            ms.type = ReflectionType::Specular;
            ms.wi = normalize(info.normal + uniform_sample_sphere(rng.random_float2()));
            ms.pdf = cosine_hemisphere_pdf(dot(info.normal, ms.wi));
            ms.f = info.mat->eval(ms.wi, -ray.dir, info.normal, info.color);

            if (ms.pdf <= 1e-5f) break;
            beta *= ms.f * abs(dot(ms.wi, info.normal)) / ms.pdf;
            specular = (ms.type == ReflectionType::Specular);
            ray = Ray(info.pos, ms.wi);

            if (depth >= 3)
            {
                float p = max(beta.x, max(beta.y, beta.z));
                if (rng.random_float() > p) break;
                beta /= p;
            }
        }
        result += L / params.spp;
    }

    int idx = iy * params.width + ix;
    params.buffer[idx] = make_float4(result, 1.0f);
}