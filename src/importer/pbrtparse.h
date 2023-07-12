#pragma once

#include "definition.h"
#include "my_math.h"
#include "scene.h"
#include "assimp.h"

class PBRTState
{
public:
    int material_id;
    float3 emission;
    Transform transform;
    shared_ptr<TriangleMesh> mesh;

public:
    PBRTState() { reset(); }

    void reset()
    {
        material_id = -1;
        emission = make_float3(0.0f);
        transform = Transform();
        mesh = nullptr;
    }
};

class PBRTParser
{
public:
    std::ifstream file;
    std::filesystem::path folder;
    AssimpImporter importer;
    shared_ptr<Scene> scene{ nullptr };
    int width{ -1 }, height{ -1 };

    PBRTState global_state;
    PBRTState attribute_state;
    bool in_attribute{ false };

private:
    string next_quoted();
    string next_bracketed();
    unordered_map<string, string> next_parameter_list();
    void ignore();  // read file but ignore the content

    shared_ptr<TriangleMesh> load_shape(const string& type, const unordered_map<string, string>& params);
    shared_ptr<Material> load_material(const string& name, const string& type, const unordered_map<string, string>& params);
    shared_ptr<Texture> load_texture(const string& name, const unordered_map<string, string>& params);
    void parse_options();
    void parse_world();

public:
    void parse(const string& filename, shared_ptr<Scene> _s);
};