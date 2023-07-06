#pragma once

#include "definition.h"
#include "my_math.h"

#ifndef SRT_HIGH_PERFORMANCE 

#define MAX_BONE_PER_VERTEX 4
class Bone
{
public:
    string name;
    Transform offset;       // offset matrix
};

#endif


class TriangleMesh
{
public:
    string name;
    int material_id{ -1 };
    AABB aabb;

    vector<float3> vertices;
    vector<uint3> indices;
    vector<float3> normals;
    vector<float3> tangents;
    vector<float2> texcoords;

#ifndef SRT_HIGH_PERFORMANCE
    bool has_bone{ false };
    vector<int> bone_ids;       // vertex number * MAX_BONE_PER_VERTEX
    vector<float> bone_weights; // vertex number * MAX_BONE_PER_VERTEX
#endif


public:
    TriangleMesh() {}
    void compute_aabb();

#ifndef SRT_HIGH_PERFORMANCE
    void reset_bones();
    void add_bone_influence(int vid, int bid, float weight);
#endif
};