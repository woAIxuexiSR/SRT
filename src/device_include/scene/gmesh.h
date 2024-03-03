#pragma once

#include "my_math.h"
#include "gmaterial.h"

class SurfaceSample
{
public:
    float3 pos;
    float3 normal;
    float2 texcoord;
    float3 color;
    Onb onb;
    GMaterial* mat;

    float pdf{ 0.0f };
};

class GTriangleMesh
{
public:
    int vert_num{ 0 }, face_num{ 0 };

    float3* vertices{ nullptr };
    uint3* indices{ nullptr };
    float3* normals{ nullptr };
    float3* tangents{ nullptr };
    float2* texcoords{ nullptr };

    float area{ 0.0f };
    float* area_cdf{ nullptr };

    GMaterial* material{ nullptr };
};

class GInstance
{
public:
    GTriangleMesh* mesh{ nullptr };
    Transform* transform{ nullptr };

public:

    // rnd: choose a face, rnd2: sample point on the face
    __device__ SurfaceSample sample(float2 rnd) const
    {
        int idx = binary_search(mesh->area_cdf, mesh->face_num, rnd.x);

        int lower = (idx == 0) ? 0 : mesh->area_cdf[idx - 1];
        rnd.x = (rnd.x - lower) / (mesh->area_cdf[idx] - lower);
        float2 uv = uniform_sample_triangle(rnd);

        uint3& index = mesh->indices[idx];
        const float3& v0 = mesh->vertices[index.x];
        const float3& v1 = mesh->vertices[index.y];
        const float3& v2 = mesh->vertices[index.z];

        float3 pos = v0 * (1.0f - uv.x - uv.y) + v1 * uv.x + v2 * uv.y;
        float2 texcoord = uv;
        if (mesh->texcoords)
            texcoord = mesh->texcoords[index.x] * (1.0f - uv.x - uv.y) + mesh->texcoords[index.y] * uv.x + mesh->texcoords[index.z] * uv.y;
        float3 color = mesh->material->surface_color(texcoord);

        float3 normal = normalize(cross(v1 - v0, v2 - v0));
        float3 shading_normal = normal;
        if (mesh->normals)
            shading_normal = mesh->normals[index.x] * (1.0f - uv.x - uv.y) + mesh->normals[index.y] * uv.x + mesh->normals[index.z] * uv.y;
        shading_normal = mesh->material->shading_normal(shading_normal, texcoord);
        if (dot(normal, shading_normal) < 0.0f)
            normal = -normal;

        float3 tangent;
        if (mesh->tangents)
            tangent = mesh->tangents[index.x] * (1.0f - uv.x - uv.y) + mesh->tangents[index.y] * uv.x + mesh->tangents[index.z] * uv.y;
        else
            tangent = (abs(shading_normal.x) > abs(shading_normal.y)) ? make_float3(0, 1, 0) : make_float3(1, 0, 0);

        return { transform->apply_point(pos), transform->apply_vector(normal), texcoord,
            color, transform->apply_onb(Onb(shading_normal, tangent)), mesh->material, 1.0f / mesh->area };
    }

    __device__ float sample_pdf() const
    {
        return 1.0f / mesh->area;
    }
};