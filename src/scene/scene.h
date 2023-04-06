#pragma once

#include <cuda_runtime.h>
#include <filesystem>
#include <set>

#include "helper_math.h"
#include "definition.h"
#include "scene/material.h"

class RenderParams
{
public:
    int width, height;
    int spp;
    string method;
    SquareMatrix<4> transform;
    float fov;
};

class ComparisonRenderParams
{
public:
    int width, height;
    int spp_1, spp_2;
    string method_1, method_2;
    SquareMatrix<4> transform;
    float fov;
};


class TriangleMesh
{
public:
    vector<float3> vertices;
    vector<uint3> indices;
    vector<float3> normals;
    vector<float2> texcoords;

    int material_id{ -1 };
    int texture_id{ -1 };

public:
    TriangleMesh() {}
    TriangleMesh(const string& type, const unordered_map<string, string>& params, SquareMatrix<4> transform);
    void create_from_ply(const string& plypath);
    void create_from_triangles(const unordered_map<string, string>& params);
};


class Texture
{
public:
    unsigned* pixels;
    uint2 resolution;

    Texture(): pixels(nullptr), resolution({ 0, 0 }) {}
    ~Texture() { if (pixels) delete[] pixels; }
    void load_from_file(const string& filename);
};


class Scene
{
public:
    vector<shared_ptr<TriangleMesh> > meshes;
    vector<shared_ptr<Texture> > textures;
    vector<shared_ptr<Material> > materials;

    unordered_map<string, int> named_materials;
    unordered_map<string, int> named_textures;
    unordered_map<int, int> mat_to_tex;

public:
    Scene() {}

    // pbrd method
    void add_mesh(const string& type, const unordered_map<string, string>& params, int material_id, SquareMatrix<4> transform);
    int add_texture(const string& type, const unordered_map<string, string>& params);
    int add_named_material(const string& name, const unordered_map<string, string>& params);
    int add_material(const string& type, const unordered_map<string, string>& params);
    int add_light_material(const string& type, const unordered_map<string, string>& params);
    int get_material_id(const string& name);

    // obj etc. method
    void load_from_model(const string& filename);

    // build from code
    void add_quad_light(float3 center, float3 normal, float3 size, float3 emission);
};