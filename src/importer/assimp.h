#pragma once

#include "definition.h"
#include "my_math.h"
#include "scene.h"

class AssimpImporter
{
private:
    std::filesystem::path folder;
    const aiScene* ascene{ nullptr };
    shared_ptr<Scene> scene{ nullptr };

    std::unordered_map<string, shared_ptr<SceneGraphNode> > scene_graph_nodes;
    std::unordered_map<string, int> texture_index_map;
    std::unordered_map<string, int> bone_index_map;

private:
    /* helper functions */

    void load_meshes();
    void load_bones(aiMesh* amesh, shared_ptr<TriangleMesh> mesh);
    int load_texture(aiMaterial* amat, aiTextureType type);
    void traverse(aiNode* anode, shared_ptr<SceneGraphNode> node, const Transform& parent_transform);
    void load_animations();

public:
    // import only mesh (for pbrt)
    shared_ptr<TriangleMesh> import_mesh(const string& filename);
    // import the total scene
    void import_scene(const string& filename, shared_ptr<Scene> _s);
};