#pragma once

#include "definition.h"
#include "my_math.h"

class TriangleMesh
{
public:
    vector<float3> vertices;
    vector<uint3> indices;
    vector<float3> normals;
    vector<float2> texcoords;

    int material_id{ -1 };
    int texture_id{ -1 };

    AABB aabb;

public:
    TriangleMesh() {}
    void set_material(int _m) { material_id = _m; }
    void set_texture(int _t) { texture_id = _t; }

    void compute_aabb();
    void apply_transform(const Transform& t);
    void load_from_ply(const string& filename);
    void load_from_others(const string& filename);
    void load_from_file(const string& filename);
    void load_from_triangles(const vector<float3>& _v, const vector<uint3>& _i, const vector<float3>& _n, const vector<float2>& _t);
};


class AnimatedTriangleMesh: public TriangleMesh
{
public:
    float start_time;
    float end_time;
    vector<Transform> transforms;

public:
    AnimatedTriangleMesh() = default;

    void set_transforms(float _s, float _e, const vector<Transform>& _t)
    {
        start_time = _s;
        end_time = _e;
        transforms = _t;
    }
};
