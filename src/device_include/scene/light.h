#pragma once

#include "my_math.h"
#include "gmesh.h"

class LightSample
{
public:
    float3 pos;
    float3 normal;      // not support shading normal
    float3 emission{ 0.0f, 0.0f, 0.0f };
    float pdf{ 0.0f };
};

class AreaLight
{
public:
    GTriangleMesh* mesh{ nullptr };
    Transform* transform{ nullptr };

    int face_num{ 0 };
    float* areas{ nullptr };
    float area_sum{ 0 };

public:
    AreaLight() {}

    __device__ LightSample sample(float2 rnd) const
    {
        float s = rnd.x * area_sum;
        float sum = 0.0f;
        for (int i = 0; i < face_num; i++)
        {
            if (s <= sum + areas[i] || i == face_num - 1)
            {
                rnd.x = clamp((s - sum) / areas[i], 0.0f, 1.0f);
                float2 p = uniform_sample_triangle(rnd);

                uint3& index = mesh->indices[i];
                const float3& v0 = mesh->vertices[index.x];
                const float3& v1 = mesh->vertices[index.y];
                const float3& v2 = mesh->vertices[index.z];

                float3 pos = v0 * (1.0f - p.x - p.y) + v1 * p.x + v2 * p.y;

                float2 texcoord = p;
                if (mesh->texcoords)
                    texcoord = mesh->texcoords[index.x] * (1.0f - p.x - p.y) + mesh->texcoords[index.y] * p.x + mesh->texcoords[index.z] * p.y;

                float3 normal;
                if (mesh->normals)
                    normal = mesh->normals[index.x] * (1.0f - p.x - p.y) + mesh->normals[index.y] * p.x + mesh->normals[index.z] * p.y;
                else
                    normal = cross(v1 - v0, v2 - v0);

                float3 emission = mesh->material->emission(texcoord);

                return { transform->apply_point(pos), transform->apply_vector(normal), emission, sample_pdf() };
            }
            sum += areas[i];
        }
        return LightSample();
    }

    // sample proportional to area
    __device__ float sample_pdf() const 
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

    __device__ float3 emission(float3 dir) const
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
    Light() {}

    __device__ LightSample sample(float2 rnd) const
    {
        float s = rnd.x * weight_sum;
        float sum = 0.0f;
        for (int i = 0; i < num; i++)
        {
            float wi = lights[i].area_sum * lights[i].mesh->material->intensity;
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
    __device__ float sample_pdf(int idx) const
    {
        return lights[idx].mesh->material->intensity / weight_sum;
    }

    __device__ float3 environment_emission(float3 dir) const
    {
        if (env_light)
            return env_light->emission(dir);
        return { 0.0f, 0.0f, 0.0f };
    }
};