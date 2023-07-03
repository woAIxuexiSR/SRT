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

    const Transform* transform = sbtData.instance->transform;
    const GTriangleMesh* mesh = sbtData.instance->mesh;

    const uint3& index = mesh->indices[prim_idx];
    const float3& v0 = mesh->vertices[index.x];
    const float3& v1 = mesh->vertices[index.y];
    const float3& v2 = mesh->vertices[index.z];

    HitInfo& prd = *(HitInfo*)getPRD<HitInfo>();
    prd.hit = true;
    prd.pos = transform->apply_point(v0 * (1.0f - uv.x - uv.y) + v1 * uv.x + v2 * uv.y);

    prd.normal = transform->apply_vector(normalize(cross(v1 - v0, v2 - v0)));
    prd.inner = false;
    if (dot(prd.normal, ray_dir) > 0.0f)
    {
        prd.normal = -prd.normal;
    }

    prd.texcoord = uv;
    if (mesh->texcoords)
        prd.texcoord = mesh->texcoords[index.x] * (1.0f - uv.x - uv.y) + mesh->texcoords[index.y] * uv.x + mesh->texcoords[index.z] * uv.y;
    float3 shading_normal = mesh->material->shading_normal(prd.normal, prd.texcoord);
    // shading_normal = transform->apply_vector(shading_normal);

    float3 tangent = shading_normal.x > 0.1f ? make_float3(0.0f, 1.0f, 0.0f) : make_float3(1.0f, 0.0f, 0.0f);
    if (mesh->tangents)
        tangent = transform->apply_vector(mesh->tangents[index.x] * (1.0f - uv.x - uv.y) + mesh->tangents[index.y] * uv.x + mesh->tangents[index.z] * uv.y);
    prd.onb = Onb(shading_normal, tangent);

    prd.color = mesh->material->surface_color(prd.texcoord);
    prd.mat = mesh->material;

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
    Light* light = params.light;

    HitInfo info; int visible;
    uint2 u = pack_pointer(&info), v = pack_pointer(&visible);

    float3 result = make_float3(0.0f);
    for (int i = 0; i < params.samples_per_pixel; i++)
    {
        float xx = (ix + rng.random_float()) / params.width;
        float yy = (iy + rng.random_float()) / params.height;
        Ray ray = camera.get_ray(xx, yy, rng);

        float3 L = make_float3(0.0f), beta = make_float3(1.0f);
        bool specular = true;
        float scatter_pdf = 1.0f;
        for (int depth = 0; depth < params.max_depth; depth++)
        {
            optixTrace(params.traversable, ray.pos, ray.dir, 1e-3f, 1e16f, 0.0f,
                OptixVisibilityMask(255), OPTIX_RAY_FLAG_DISABLE_ANYHIT,
                RADIANCE_RAY_TYPE, RAY_TYPE_COUNT, RADIANCE_RAY_TYPE,
                u.x, u.y);

            // miss
            if (!info.hit)
            {
                L += beta * light->environment_emission(ray.dir);
                break;
            }

            // hit light
            if (info.hit && info.mat->is_emissive())
            {
                if (info.inner) break;

                float mis_weight = 1.0f;
                if (params.use_nee && !specular)
                {
                    mis_weight = 0.0f;
                    if (params.use_mis)
                    {
                        float t2 = dot(info.pos - ray.pos, info.pos - ray.pos);
                        float light_pdf = light->sample_pdf(info.light_id) * t2 / dot(info.normal, -ray.dir);
                        mis_weight = scatter_pdf / (light_pdf + scatter_pdf);
                    }
                }
                L += beta * info.mat->emission(info.texcoord) * mis_weight;
                break;
            }

            // next event estimation
            if (params.use_nee && !info.mat->is_specular())
            {
                LightSample ls = light->sample(rng.random_float2());
                Ray shadow_ray(info.pos, normalize(ls.pos - info.pos));

                float cos_i = dot(info.normal, shadow_ray.dir);
                float cos_light = dot(ls.normal, -shadow_ray.dir);
                if ((cos_light > 0.0f) && (cos_i > 0.0f || info.mat->is_transmissive()))
                {
                    float t = length(ls.pos - info.pos);
                    optixTrace(params.traversable, shadow_ray.pos, shadow_ray.dir, 1e-3f, t - 1e-3f, 0.0f,
                        OptixVisibilityMask(255), OPTIX_RAY_FLAG_DISABLE_ANYHIT,
                        SHADOW_RAY_TYPE, RAY_TYPE_COUNT, SHADOW_RAY_TYPE,
                        v.x, v.y);

                    if (visible)
                    {
                        float mis_weight = 1.0f;
                        float light_pdf = ls.pdf * t * t / cos_light;
                        if (params.use_mis)
                        {
                            float mat_pdf = info.mat->sample_pdf(shadow_ray.dir, -ray.dir, info.onb, info.color);
                            mis_weight = light_pdf / (light_pdf + mat_pdf);
                        }
                        L += beta * info.mat->eval(shadow_ray.dir, -ray.dir, info.onb, info.color)
                            * abs(cos_i) * ls.emission * mis_weight / light_pdf;
                    }
                }
            }

            // sample next direction
            BxDFSample ms = info.mat->sample(-ray.dir, rng.random_float2(), info.onb, info.color);
            if (ms.pdf <= 1e-5f) break;
            beta *= ms.f * abs(dot(ms.wi, info.normal)) / ms.pdf;

            specular = info.mat->is_specular();
            ray = Ray(info.pos, ms.wi);
            scatter_pdf = ms.pdf;

            // russian roulette
            if (depth >= params.rr_depth)
            {
                float p = max(beta.x, max(beta.y, beta.z));
                if (rng.random_float() > p) break;
                beta /= p;
            }
        }

        result += L / params.samples_per_pixel;
    }

    params.pixels[idx] = make_float4(result, 1.0f);
}