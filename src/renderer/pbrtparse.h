#pragma once

#include "definition.h"
#include "helper_math.h"
#include "scene.h"


class PBRTParser
{
public:    int width, height;
private:
    std::ifstream file;
    std::filesystem::path folderpath;
    shared_ptr<Scene> scene;

    // current state
    int material_id{ -1 };
    int texture_id{ -1 };
    float3 emission{ 0.0f, 0.0f,0.0f };
    Transform transform;

private:
    // helper functions

    void reset_state();
    string next_quoted();
    string next_bracketed();
    unordered_map<string, string> next_parameter_list();
    void ignore();

    void load_shape(const string& type, const unordered_map<string, string>& params);
    void load_material(const string& name, const string& type, const unordered_map<string, string>& params);
    void load_texture(const string& name, const unordered_map<string, string>& params);

public:
    PBRTParser(const string& filename);
    ~PBRTParser();

    void parse();
    shared_ptr<Scene> get_scene() const { return scene; }
};

