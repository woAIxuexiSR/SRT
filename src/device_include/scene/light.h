#pragma once

#include "my_math.h"
#include "gmesh.h"

class LightSample
{
public:
    float3 pos;
    float3 normal;
    float3 shading_normal;
    float3 emission;
    float pdf;
};

class AreaLight
{
public:
    int face_num { 0 };
    GInstance* instance{ nullptr };
    float* areas{ nullptr };
    float area_sum{ 0 };

public:
    AreaLight() {}
    AreaLight(int _n, GInstance* _m, float* _a, float _s)
        : face_num(_n), instance(_m), areas(_a), area_sum(_s) {}

    __device__ LightSample sample(float2 rnd)
    {
        float s = rnd.x * area_sum;
        float sum = 0.0f;
        for (int i = 0; i < face_num; i++)
        {
            if (s <= sum + areas[i] || i == face_num - 1)
            {
                rnd.x = clamp((s - sum) / areas[i], 0.0f, 1.0f);
                float2 p = uniform_sample_triangle(rnd);

                GTriangleMesh* mesh = instance->mesh;
                uint3& index = mesh->indices[i];

                float3 pos = mesh->vertices[index.x] * (1.0f - p.x - p.y) + mesh->vertices[index.y] * p.x + mesh->vertices[index.z] * p.y;

                float3 normal;
                if (mesh->normals)
                    normal = mesh->normals[index.x] * (1.0f - p.x - p.y) + mesh->normals[index.y] * p.x + mesh->normals[index.z] * p.y;
                else
                    normal = cross(mesh->vertices[index.y] - mesh->vertices[index.x], mesh->vertices[index.z] - mesh->vertices[index.x]);

                float2 texcoord;
                if (mesh->texcoords)
                    texcoord = mesh->texcoords[index.x] * (1.0f - p.x - p.y) + mesh->texcoords[index.y] * p.x + mesh->texcoords[index.z] * p.y;

                float3 shading_normal = mesh->material->shading_normal(normal, texcoord);
                float3 emission = mesh->material->emission(texcoord);

                Transform* t = instance->transform;
                return { t->apply_point(pos), t->apply_vector(normal), t->apply_vector(shading_normal), emission, sample_pdf() };
            }
            sum += areas[i];
        }
        return LightSample();
    }

    // sample proportional to area
    __device__ float sample_pdf()
    {
        return 1.0f / area_sum;
    }
};


class EnvironmentLight
{
public:
    enum class Type { Constant, UVMap };

    Type type{ Type::Constant };
    float3 emission_color{ 0.0f, 0.0f, 0.0f };
    cudaTextureObject_t texture{ 0 };

public:
    EnvironmentLight() {}
    EnvironmentLight(Type _t, float3 _c, cudaTextureObject_t _tex = 0)
        : type(_t), emission_color(_c), texture(_tex) {}

    __device__ float3 emission(float3 dir)
    {
        switch (type)
        {
        case Type::Constant:
            return emission_color;
        case Type::UVMap:
        {
            // world space y axis is up, math space z axis is up
            float3 math_space_dir = make_float3(-dir.x, dir.z, dir.y);
            float2 phi_theta = cartesian_to_spherical_uv(math_space_dir);
            float2 uv = make_float2(phi_theta.x * 0.5f * (float)M_1_PI + 0.5f, phi_theta.y * (float)M_1_PI);
            return make_float3(tex2D<float4>(texture, uv.x, uv.y));
        }
        default:
            return { 0.0f, 0.0f, 0.0f };
        }
    }
};

// many area lights and one environment light
class Light
{
public:
    int num;
    AreaLight* lights{ nullptr };
    float weight_sum{ 0.0f };

    EnvironmentLight* env_light{ nullptr };

public:
    Light(int _n, AreaLight* _l, float _s, EnvironmentLight* _e = nullptr)
        : num(_n), lights(_l), weight_sum(_s), env_light(_e) {}

    __device__ LightSample sample(float2 rnd)
    {
        float s = rnd.x * weight_sum;
        float sum = 0.0f;
        for (int i = 0; i < num; i++)
        {
            float wi = lights[i].area_sum * lights[i].instance->mesh->material->intensity;
            if (s <= sum + wi || i == num - 1)
            {
                rnd.x = clamp((s - sum) / wi, 0.0f, 1.0f);
                LightSample ls = lights[i].sample(rnd);
                ls.pdf *= wi / weight_sum;
                return ls;
            }
            sum += wi;
        }
        return LightSample();
    }

    // sample proportional to area * intensity
    __device__ float sample_pdf(int idx)
    {
        return lights[idx].instance->mesh->material->intensity / weight_sum;
    }

    __device__ float3 environment_emission(float3 dir)
    {
        if (env_light)
            return env_light->emission(dir);
        return { 0.0f, 0.0f, 0.0f };
    }
};