#pragma once

#include "definition.h"
#include "my_math.h"
#include "helper_cuda.h"

#include "animation.h"
#include "texture.h"
#include "material.h"
#include "mesh.h"

#include "scene/camera.h"
#include "scene/gmaterial.h"
#include "scene/gmesh.h"
#include "scene/light.h"

class SceneGraphNode
{
public:
    string name;
    Transform transform;        // local transform
    int animation_id{ -1 };

    vector<int> instance_ids;
    int bone_id{ -1 };

    vector<shared_ptr<SceneGraphNode> > children;
};

class GScene
{
public:
    vector<GPUMemory<float3> > vertex_buffer;
    vector<GPUMemory<uint3> > index_buffer;
    vector<GPUMemory<float3> > normal_buffer;
    vector<GPUMemory<float3> > tangent_buffer;
    vector<GPUMemory<float2> > texcoord_buffer;

    vector<GPUMemory<float3> > animated_vertex_buffer;
    vector<GPUMemory<float3> > animated_normal_buffer;
    vector<GPUMemory<float3> > animated_tangent_buffer;
    vector<GPUMemory<int> > bone_id_buffer;
    vector<GPUMemory<float> > bone_weight_buffer;
    GPUMemory<Transform> bone_transform_buffer;

    GPUMemory<GTriangleMesh> mesh_buffer;
    GPUMemory<GMaterial> material_buffer;
    vector<cudaArray_t> texture_arrays;
    vector<cudaTextureObject_t> texture_objects;

    GPUMemory<GInstance> instance_buffer;
    GPUMemory<Transform> instance_transform_buffer;
    vector<GPUMemory<int> > instance_meshid_buffer;

    GPUMemory<Light> light_buffer;
    GPUMemory<AreaLight> area_light_buffer;
    vector<int> instance_light_id;
    vector<GPUMemory<float> > light_area_buffer;
    GPUMemory<EnvironmentLight> environment_light_buffer;
};


class Scene
{
public:
    shared_ptr<Camera> camera;
    float3 background{ 0.0f, 0.0f, 0.0f };
    int environment_map_id{ -1 };
    AABB aabb;              // not considering animation

    vector<shared_ptr<TriangleMesh> > meshes;
    vector<Bone> bones;
    vector<Transform> bone_transforms;
    vector<int> instances;  // mesh id
    vector<Transform> instance_transforms;
    vector<shared_ptr<Material> > materials;
    vector<shared_ptr<Texture> > textures;
    vector<shared_ptr<Animation> > animations;

    shared_ptr<SceneGraphNode> root;
    GScene gscene;

public:
    Scene() {}

    /* build functions  */

    int add_material(shared_ptr<Material> material);
    int find_material(const string& name);
    int add_texture(shared_ptr<Texture> texture);
    int find_texture(const string& name);
    int add_animation(shared_ptr<Animation> animation);

    void set_camera(shared_ptr<Camera> _c) { camera = _c; }
    void set_background(float3 _b) { background = _b; }
    void set_environment_map(shared_ptr<Texture> _t) { environment_map_id = add_texture(_t); }

    /* GPU build functions */

    void build_gscene();
    void build_gscene_meshes();
    void build_gscene_instances();
    void build_gscene_materials();
    void build_gscene_textures();
    void build_gscene_lights();


    /* useful functions */

    void compute_aabb();

    void update(float t);
    void update_node(shared_ptr<SceneGraphNode> node, float t, const Transform& parent_transform);
    void update_gscene();

    void render_ui();

};
