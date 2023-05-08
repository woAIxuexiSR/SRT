#include <optix_device.h>
#include <device_launch_parameters.h>

#include "helper_optix.h"
#include "my_params.h"
#include "my_math.h"

extern "C" __constant__ PathTracerParams params;

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
    prd.inner = false;
    if (dot(prd.normal, ray_dir) > 0.0f)
    {
        prd.normal = -prd.normal;
        prd.inner = true;
    }

    if (sbtData.has_texture && sbtData.texcoord)
    {
        float2 tc = sbtData.texcoord[index.x] * (1.0f - uv.x - uv.y) + sbtData.texcoord[index.y] * uv.x + sbtData.texcoord[index.z] * uv.y;
        float4 tex = tex2D<float4>(sbtData.texture, tc.x, tc.y);
        prd.color = make_float3(tex.x, tex.y, tex.z);
    }

    prd.light_id = sbtData.light_id;
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

    RandomGenerator rng(params.seed + idx, 0);
    Camera& camera = params.camera;
    Light& light = params.extra.light;

    HitInfo info; int visible;
    thrust::pair<unsigned, unsigned> u = pack_pointer(&info), v = pack_pointer(&visible);

    float3 result = make_float3(0.0f);
    for (int i = 0; i < params.extra.spp; i++)
    {
        float xx = (ix + rng.random_float()) / params.width;
        float yy = (iy + rng.random_float()) / params.height;
        Ray ray = camera.get_ray(xx, yy);

        float3 L = make_float3(0.0f), beta = make_float3(1.0f);
        bool specular = true;
        float scatter_pdf = 1.0f;
        float cos_theta = 1.0f;
        for (int depth = 0; depth < params.extra.max_depth; depth++)
        {
            optixTrace(params.traversable, ray.pos, ray.dir, 1e-3f, 1e16f, 0.0f,
                OptixVisibilityMask(255), OPTIX_RAY_FLAG_DISABLE_ANYHIT,
                RADIANCE_RAY_TYPE, RAY_TYPE_COUNT, RADIANCE_RAY_TYPE,
                u.first, u.second);

            if (!info.hit)
            {
                L += beta * light.environment_emission(ray.dir);
                break;
            }

            if (info.hit && info.mat->is_emissive())
            {
                if (info.inner)
                    break;
                if (specular)
                    L += beta * info.mat->get_emission_color() * info.mat->get_intensity();
                else
                {
                    float t2 = dot(info.pos - ray.pos, info.pos - ray.pos);
                    float light_pdf = light.sample_pdf(info.light_id) * t2 / dot(info.normal, -ray.dir);
                    // float mat_pdf = scatter_pdf;
                    float mat_pdf = cosine_hemisphere_pdf(cos_theta);
                    float mis_weight = mat_pdf / (light_pdf + mat_pdf);
                    L += beta * info.mat->get_emission_color() * info.mat->get_intensity() * mis_weight;
                }
                break;
            }

            if (!info.mat->is_specular())
            {
                LightSample ls = light.sample(rng.random_float2());
                Ray shadow_ray(info.pos, normalize(ls.pos - info.pos));

                float cos_i = dot(info.normal, shadow_ray.dir);
                float cos_light = dot(ls.normal, -shadow_ray.dir);
                if ((cos_light > 0.0f) && (cos_i > 0.0f || info.mat->is_transmissive()))
                {
                    float t = length(ls.pos - info.pos);
                    optixTrace(params.traversable, shadow_ray.pos, shadow_ray.dir, 1e-3f, t - 1e-3f, 0.0f,
                        OptixVisibilityMask(255), OPTIX_RAY_FLAG_DISABLE_ANYHIT,
                        SHADOW_RAY_TYPE, RAY_TYPE_COUNT, SHADOW_RAY_TYPE,
                        v.first, v.second);

                    if (visible)
                    {
                        float light_pdf = ls.pdf * t * t / cos_light;
                        // float mat_pdf = info.mat->sample_pdf(shadow_ray.dir, -ray.dir, info.normal, info.color, info.inner);
                        float mat_pdf = cosine_hemisphere_pdf(dot(info.normal, shadow_ray.dir));
                        float mis_weight = light_pdf / (light_pdf + mat_pdf);
                        L += beta * info.mat->eval(shadow_ray.dir, -ray.dir, info.normal, info.color, info.inner)
                            * abs(cos_i) * ls.emission * mis_weight / light_pdf;
                    }
                }
            }

            MaterialSample ms = info.mat->sample(-ray.dir, info.normal, rng.random_float2(), info.color, info.inner);

            if (ms.pdf <= 1e-5f) break;
            beta *= ms.f * abs(dot(ms.wi, info.normal)) / ms.pdf;
            cos_theta = dot(ms.wi, info.normal);
            specular = (ms.type == ReflectionType::Specular);
            ray = Ray(info.pos, ms.wi);
            scatter_pdf = ms.pdf;

            if (depth >= 3)
            {
                float p = max(beta.x, max(beta.y, beta.z));
                if (rng.random_float() > p) break;
                beta /= p;
            }
        }
        result += L / params.extra.spp;
    }

    params.pixels[idx] = make_float4(result, 1.0f);
}