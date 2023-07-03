#pragma once

#include "definition.h"
#include "my_math.h"
#include "scene.h"

class AssimpImporter
{
private:
    std::filesystem::path folder;
    const aiScene* ascene {nullptr};
    shared_ptr<Scene> scene {nullptr};
    std::unordered_map<string, shared_ptr<SceneGraphNode> > scene_graph_nodes;

public:
    void import(const string& filename, shared_ptr<Scene> _scene);
    void load_meshes();
    // void load_bones(aiMesh* amesh, shared_ptr<Mesh> mesh);
    void traverse(aiNode* anode, shared_ptr<SceneGraphNode> node);
    void load_animations();
};