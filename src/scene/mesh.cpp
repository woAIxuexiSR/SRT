#include "mesh.h"

void TriangleMesh::compute_area()
{
    area_cdf.resize(indices.size());
    for (int i = 0; i < indices.size(); i++)
    {
        float3 v0 = vertices[indices[i].x];
        float3 v1 = vertices[indices[i].y];
        float3 v2 = vertices[indices[i].z];

        float A = 0.5f * length(cross(v1 - v0, v2 - v0));
        area += A;
        area_cdf[i] = area;
    }
    for (int i = 0; i < area_cdf.size(); i++)
        area_cdf[i] /= area;
}

void TriangleMesh::compute_aabb()
{
    assert(!vertices.empty());
    aabb = AABB();
    for (auto v : vertices)
        aabb.expand(v);
}

void TriangleMesh::reset_bones()
{
    has_bone = true;
    bone_ids.resize(vertices.size() * MAX_BONE_PER_VERTEX, -1);
    bone_weights.resize(vertices.size() * MAX_BONE_PER_VERTEX, 0);
}

void TriangleMesh::add_bone_influence(int vid, int bid, float weight)
{
    assert(vid < vertices.size());
    int idx = vid * MAX_BONE_PER_VERTEX;
    for (int i = 0; i < MAX_BONE_PER_VERTEX; i++)
    {
        if (bone_ids[idx + i] == -1)
        {
            bone_ids[idx + i] = bid;
            bone_weights[idx + i] = weight;
            return;
        }
    }
    cout << "ERROR::Too many bone influences for vertex " << vid << endl;
}