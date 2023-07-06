#include "mesh.h"

void TriangleMesh::compute_aabb()
{
    assert(!vertices.empty());
    aabb = AABB();
    for (auto v : vertices)
        aabb.expand(v);
}

#ifndef SRT_HIGH_PERFORMANCE

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

#endif