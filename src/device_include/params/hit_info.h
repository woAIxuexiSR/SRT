#pragma once

#include "my_math.h"
#include "scene/gmesh.h"
#include "scene/gmaterial.h"

class HitInfo
{
public:
    bool hit;

    float3 pos;
    float3 normal;      // not guaranteed to be the same side as ray
    float2 texcoord;
    float3 color;

    Onb onb;            // build from shading normal
    const GMaterial* mat;
    int light_id;
};

__device__ inline bool check_valid(float p)
{
    return !isnan(p) && !isinf(p);
}

__device__ inline void get_hitinfo(HitInfo& info, const GTriangleMesh* mesh, const Transform* T,
    const int prim_idx, const float2 uv, const float3 ray_dir, const int light_id)
{
    const uint3& index = mesh->indices[prim_idx];
    const float3& v0 = mesh->vertices[index.x];
    const float3& v1 = mesh->vertices[index.y];
    const float3& v2 = mesh->vertices[index.z];

    float3 pos = v0 * (1.0f - uv.x - uv.y) + v1 * uv.x + v2 * uv.y;
    float3 normal;
    if (mesh->normals)
        normal = normalize(mesh->normals[index.x] * (1.0f - uv.x - uv.y) + mesh->normals[index.y] * uv.x + mesh->normals[index.z] * uv.y);
    else
        normal = normalize(cross(v1 - v0, v2 - v0));
    float2 texcoord = uv;
    if (mesh->texcoords)
        texcoord = mesh->texcoords[index.x] * (1.0f - uv.x - uv.y) + mesh->texcoords[index.y] * uv.x + mesh->texcoords[index.z] * uv.y;
    float3 color = mesh->material->surface_color(texcoord);

    float3 shading_normal = mesh->material->shading_normal(normal, texcoord);
    float3 tangent;
    if (mesh->tangents)
        tangent = mesh->tangents[index.x] * (1.0f - uv.x - uv.y) + mesh->tangents[index.y] * uv.x + mesh->tangents[index.z] * uv.y;
    else
        tangent = (abs(shading_normal.x) > abs(shading_normal.y)) ? make_float3(0, 1, 0) : make_float3(1, 0, 0);

    info.hit = true;
    info.pos = T->apply_point(pos);
    info.normal = T->apply_vector(normal);
    info.texcoord = texcoord;
    info.color = color;
    info.onb = T->apply_onb(Onb(shading_normal, tangent));
    info.mat = mesh->material;
    info.light_id = light_id;
}