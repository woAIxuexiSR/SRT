#pragma once

#include "definition.h"
#include "my_math.h"

#include "scene/camera.h"
#include "scene/material.h"
#include "mesh.h"
#include "texture.h"

void set_material_property(shared_ptr<Material> material, const string& name, float value);

class Scene
{
public:
    shared_ptr<Camera> camera;

    vector<shared_ptr<TriangleMesh> > meshes;
    vector<shared_ptr<AnimatedTriangleMesh> > animated_meshes;
    AABB aabb;

    vector<shared_ptr<Material> > materials;
    vector<string> material_names;
    vector<shared_ptr<Texture> > textures;
    vector<string> texture_names;
    unordered_map<int, int> material_to_texture;

    float3 background{ 0.0f, 0.0f, 0.0f };
    vector<shared_ptr<Texture> > environment_map;

public:
    Scene() {}
    bool is_static() const { return animated_meshes.empty(); }
    AABB get_aabb() const { return aabb; }
    void set_camera(shared_ptr<Camera> _c) { camera = _c; }

    int add_mesh(shared_ptr<TriangleMesh> mesh, int material_id);
    int add_animated_mesh(shared_ptr<AnimatedTriangleMesh> mesh, int material_id);
    int add_material(shared_ptr<Material> material, string name = "", int texture_id = -1);
    int add_textures(shared_ptr<Texture> texture, string name = "");

    void set_background(float3 _b) { background = _b; }
    void load_environment_map(const vector<string>& faces);

    int get_material_id(const string& name) const;
    int get_texture_id(const string& name) const;

    // obj etc. model
    void load_from_model(const string& filename);

    // pbrt scene
    void load_from_pbrt(const string& filename);

    // config
    void load_from_config(const json& config, int width, int height);

    // ui
    void render_ui();
};

