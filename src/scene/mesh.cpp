#include "mesh.h"

TriangleMesh::TriangleMesh(const vector<float3>& _v, const vector<uint3>& _i,
    const vector<float3>& _n, const vector<float3>& _t, const vector<float2>& _tc)
    : vertices(_v), indices(_i), normals(_n), tangents(_t), texcoords(_tc)
{
    bone_ids.resize(vertices.size() * MAX_BONE_PER_VERTEX, -1);
    bone_weights.resize(vertices.size() * MAX_BONE_PER_VERTEX, 0.0f);
    compute_aabb();
}

void TriangleMesh::compute_aabb()
{
    assert(!vertices.empty());
    aabb = AABB();
    for (auto v : vertices)
        aabb.expand(v);
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