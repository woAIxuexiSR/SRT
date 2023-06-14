#pragma once

#include "definition.h"
#include "my_math.h"
#include "bone.h"
#include "animation.h"

// triangle mesh
class Mesh
{
public:
    enum class MotionType { Static, Rigid, Skeletal };

    MotionType motion_type { MotionType::Static };
    
    vector<float3> vertices;
    vector<uint3> indices;
    vector<float3> normals;
    vector<float3> tangents;
    vector<float2> texcoords;
    vector<int> bone_ids;       // vertex number * MAX_BONE_PER_VERTEX
    vector<float> bone_weights; // vertex number * MAX_BONE_PER_VERTEX

    int material_id{-1};
    AABB aabb;

public:
    Mesh() {}

    void load_from_triangles(const vector<float3>& _v, const vector<uint3>& _i, const vector<float3>& _n, const vector<float3>& _t, const vector<float2>& _tc)
    {
        vertices = _v;
        indices = _i;
        normals = _n;
        tangents = _t;
        texcoords = _tc;

        int num_vertices = vertices.size();
        bone_ids.resize(num_vertices * MAX_BONE_PER_VERTEX, -1);
        bone_weights.resize(num_vertices * MAX_BONE_PER_VERTEX, 0.0f);
    }
};