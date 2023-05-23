#pragma once

#include "definition.h"
#include "my_math.h"
#include "helper_cuda.h"
#include "helper_scene.h"

#include "scene/camera.h"
#include "scene/light.h"
#include "scene/material.h"
#include "mesh.h"
#include "texture.h"

class DeviceSceneData
{
public:
    // meshes
    vector<GPUMemory<float3> > vertex_buffer;
    vector<GPUMemory<uint3> > index_buffer;
    vector<GPUMemory<float3> > normal_buffer;
    vector<GPUMemory<float2> > texcoord_buffer;

    // materials
    GPUMemory<Material> material_buffer;

    // textures
    vector<cudaArray_t> texture_arrays;
    vector<cudaTextureObject_t> texture_objects;

    // light 
    Light light;

    GPUMemory<DiffuseAreaLight> light_buffer;
    vector<GPUMemory<float> > light_area_buffer;
    vector<int> meshid_to_lightid;

    GPUMemory<EnvironmentLight> environment_light_buffer;
    cudaArray_t environment_map_array;
    cudaTextureObject_t environment_map_object;
};

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

    DeviceSceneData d_scene;

public:
    Scene() {}
    bool is_static() const { return animated_meshes.empty() && !camera->moved; }
    AABB get_aabb() const { return aabb; }
    void set_camera(shared_ptr<Camera> _c) { camera = _c; }

    int add_mesh(shared_ptr<TriangleMesh> mesh, int material_id);
    int add_animated_mesh(shared_ptr<AnimatedTriangleMesh> mesh, int material_id);
    int add_material(shared_ptr<Material> material, string name = "", int texture_id = -1);
    int add_texture(shared_ptr<Texture> texture, string name = "");

    void set_background(float3 _b) { background = _b; }
    void load_environment_map(const vector<string>& faces);

    int get_material_id(const string& name) const;
    int get_texture_id(const string& name) const;

    // obj etc. model
    void load_from_model(const string& filename);

    // build device data
    void build_device_data();
    void create_device_meshes();
    void create_device_materials();
    void create_device_textures();
    void create_device_environment_map();
    void create_device_lights();

    // ui
    void render_ui();
};

