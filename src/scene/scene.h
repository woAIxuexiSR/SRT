#pragma once

#include "definition.h"
#include "my_math.h"
#include "texture.h"
#include "material.h"
#include "mesh.h"

class SceneGraphNode
{
public:
    string name;
    Transform transform;
    shared_ptr<Animation> animation{nullptr};

    vector<shared_ptr<SceneGraphNode> > children;
    vector<int> mesh_ids;
    int bone_id {-1};
};

class Scene
{
public:
    // shared_ptr<Camera> camera;
    float3 background {0.0f, 0.0f, 0.0f};
    int environment_map{-1};
    AABB aabb;

    vector<shared_ptr<Mesh> > meshes;
    vector<Material> materials;
    vector<Texture> textures;
    vector<Bone> bones;

    shared_ptr<SceneGraphNode> root;

public:
    Scene() {}

    void update();
};