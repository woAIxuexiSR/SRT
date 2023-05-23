#pragma once

#include "definition.h"
#include "helper_math.h"
#include "scene.h"

class PBRTState
{
public:
    int material_id;
    float3 emission;
    Transform transform;
    shared_ptr<TriangleMesh> mesh;

    void reset();
};


class PBRTParser
{
private:
    std::ifstream file;
    std::filesystem::path folderpath;

    PBRTState global_state;
    PBRTState attribute_state;
    bool in_attribute;

public:
    int width, height;
    shared_ptr<Scene> scene;

private:
    // helper functions
    string next_quoted();
    string next_bracketed();
    unordered_map<string, string> next_parameter_list();
    void ignore();

    shared_ptr<TriangleMesh> load_shape(const string& type, const unordered_map<string, string>& params);
    void load_material(const string& name, const string& type, const unordered_map<string, string>& params);
    void load_texture(const string& name, const unordered_map<string, string>& params);

public:
    PBRTParser(const string& filename);
    ~PBRTParser();

    void parse();
};