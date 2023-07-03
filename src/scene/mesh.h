#pragma once

#include "definition.h"
#include "my_math.h"

#define MAX_BONE_PER_VERTEX 4

class Bone
{
public:
    string name;
    Transform offset;       // offset matrix
};


class TriangleMesh
{
public:
    vector<float3> vertices;
    vector<uint3> indices;
    vector<float3> normals;
    vector<float3> tangents;
    vector<float2> texcoords;

    bool has_bone{ false };
    vector<int> bone_ids;       // vertex number * MAX_BONE_PER_VERTEX
    vector<float> bone_weights; // vertex number * MAX_BONE_PER_VERTEX

    string name;
    int material_id{ -1 };
    AABB aabb;

public:
    TriangleMesh() {}
    TriangleMesh(const vector<float3>& _v, const vector<uint3>& _i,
        const vector<float3>& _n, const vector<float3>& _t, const vector<float2>& _tc);

    void compute_aabb();
    void add_bone_influence(int vid, int bid, float weight);
};