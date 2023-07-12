#include "assimp.h"

float3 ai_convert(const aiVector3D& v) { return make_float3(v.x, v.y, v.z); }
float3 ai_convert(const aiColor3D& c) { return make_float3(c.r, c.g, c.b); }
float2 ai_convert(const aiVector2D& v) { return make_float2(v.x, v.y); }
Quaternion ai_convert(const aiQuaternion& q) { return Quaternion(q.x, q.y, q.z, q.w); }
string ai_convert(const aiString& s) { return string(s.C_Str()); }
Transform ai_convert(const aiMatrix4x4& m)
{
    return Transform(SquareMatrix<4>(
        m.a1, m.a2, m.a3, m.a4,
        m.b1, m.b2, m.b3, m.b4,
        m.c1, m.c2, m.c3, m.c4,
        m.d1, m.d2, m.d3, m.d4
    ));
}
shared_ptr<Material> ai_convert(aiMaterial* amat)
{
    aiColor3D diffuse_color{ 0,0,0 };
    aiColor3D specular_color{ 0,0,0 };
    aiColor3D emission{ 0,0,0 };
    float emissiveStrength = 1.0f;
    float ior = 1.5f;
    float metallic = 0.0f;
    float subsurface = 0.0f;
    float roughness = 0.5f;
    float specular = 0.5f;
    float specularTint = 0.0f;
    float anisotropic = 0.0f;
    float sheen = 0.0f;
    float sheenTint = 0.0f;
    float clearcoat = 0.0f;
    float clearcoatGloss = 0.0f;
    float specTrans = 0.0f;

    amat->Get(AI_MATKEY_COLOR_DIFFUSE, diffuse_color);
    amat->Get(AI_MATKEY_COLOR_SPECULAR, specular_color);
    amat->Get(AI_MATKEY_COLOR_EMISSIVE, emission);
    amat->Get(AI_MATKEY_EMISSIVE_INTENSITY, emissiveStrength);
    amat->Get(AI_MATKEY_REFRACTI, ior);
    amat->Get(AI_MATKEY_METALLIC_FACTOR, metallic);
    amat->Get(AI_MATKEY_VOLUME_THICKNESS_FACTOR, subsurface);
    amat->Get(AI_MATKEY_ROUGHNESS_FACTOR, roughness);
    amat->Get(AI_MATKEY_SPECULAR_FACTOR, specular);
    amat->Get(AI_MATKEY_GLOSSINESS_FACTOR, specularTint);
    amat->Get(AI_MATKEY_ANISOTROPY_FACTOR, anisotropic);
    amat->Get(AI_MATKEY_SHEEN_COLOR_FACTOR, sheen);
    amat->Get(AI_MATKEY_SHEEN_ROUGHNESS_FACTOR, sheenTint);
    amat->Get(AI_MATKEY_CLEARCOAT_FACTOR, clearcoat);
    amat->Get(AI_MATKEY_CLEARCOAT_ROUGHNESS_FACTOR, clearcoatGloss);
    amat->Get(AI_MATKEY_TRANSMISSION_FACTOR, specTrans);

    float3 emission_color = ai_convert(emission) * emissiveStrength;
    emissiveStrength = length(emission_color);
    emission_color = emissiveStrength > 0 ? normalize(emission_color) : emission_color;
    float3 dc = ai_convert(diffuse_color), sc = ai_convert(specular_color);
    float3 base_color = dot(dc, dc) > 0 ? dc : sc;

    shared_ptr<Material> material = make_shared<Material>();
    material->name = ai_convert(amat->GetName());
    material->base_color = base_color;
    material->emission_color = emission_color;
    material->intensity = emissiveStrength;
    material->bxdf.type = BxDF::Type::Disney;
    material->bxdf.ior = ior;
    material->bxdf.metallic = metallic;
    material->bxdf.subsurface = subsurface;
    material->bxdf.roughness = max(roughness, 0.001f);
    material->bxdf.specular = specular;
    material->bxdf.specularTint = specularTint;
    material->bxdf.anisotropic = anisotropic;
    material->bxdf.sheen = sheen;
    material->bxdf.sheenTint = sheenTint;
    material->bxdf.clearcoat = clearcoat;
    material->bxdf.clearcoatGloss = max(clearcoatGloss, 0.001f);
    material->bxdf.specTrans = specTrans;

    return material;
}

void AssimpImporter::load_meshes()
{
    for (int i = 0; i < ascene->mNumMeshes; i++)
    {
        aiMesh* amesh = ascene->mMeshes[i];

        vector<float3> vertices;
        vector<uint3> indices;
        vector<float3> normals;
        vector<float3> tangents;
        vector<float2> texcoords;
        for (int j = 0; j < amesh->mNumVertices; j++)
        {
            vertices.push_back(ai_convert(amesh->mVertices[j]));
            if (amesh->HasNormals())
                normals.push_back(ai_convert(amesh->mNormals[j]));
            if (amesh->HasTangentsAndBitangents())
                tangents.push_back(ai_convert(amesh->mTangents[j]));
            if (amesh->HasTextureCoords(0))
            {
                float3 texcoord = ai_convert(amesh->mTextureCoords[0][j]);
                texcoords.push_back({ texcoord.x, texcoord.y });
            }
        }
        for (int j = 0; j < amesh->mNumFaces; j++)
        {
            aiFace face = amesh->mFaces[j];
            assert(face.mNumIndices == 3);
            indices.push_back({ face.mIndices[0], face.mIndices[1], face.mIndices[2] });
        }
        string name = ai_convert(amesh->mName);

        shared_ptr<TriangleMesh> mesh =
            make_shared<TriangleMesh>(name, vertices, indices, normals, tangents, texcoords);

        // load_bone
        load_bones(amesh, mesh);

        // load material
        aiMaterial* amat = ascene->mMaterials[amesh->mMaterialIndex];
        shared_ptr<Material> material = ai_convert(amat);

        // load texture
        material->color_tex_id = load_texture(amat, aiTextureType_DIFFUSE);
        material->normal_tex_id = load_texture(amat, aiTextureType_NORMALS);

        // add all to scene
        int material_id = scene->add_material(material);
        mesh->material_id = material_id;
        mesh->compute_aabb();
        scene->meshes.push_back(mesh);
    }
}

void AssimpImporter::load_bones(aiMesh* amesh, shared_ptr<TriangleMesh> mesh)
{
    if (!amesh->HasBones())
        return;

    mesh->reset_bones();
    for (int i = 0; i < amesh->mNumBones; i++)
    {
        string name = ai_convert(amesh->mBones[i]->mName);

        int id = -1;
        auto it = bone_index_map.find(name);
        if (it != bone_index_map.end())
            id = it->second;
        else
        {
            Bone bone;
            bone.name = name;
            bone.offset = ai_convert(amesh->mBones[i]->mOffsetMatrix);
            id = scene->add_bone(bone);
            bone_index_map[name] = id;
        }

        aiVertexWeight* weights = amesh->mBones[i]->mWeights;
        int weights_num = amesh->mBones[i]->mNumWeights;
        for (int j = 0; j < weights_num; j++)
        {
            int vid = weights[j].mVertexId;
            float weight = weights[j].mWeight;
            assert(vid < mesh->vertices.size());
            mesh->add_bone_influence(vid, id, weight);
        }
    }
}

int AssimpImporter::load_texture(aiMaterial* amat, aiTextureType type)
{
    aiString texname;
    if (amat->GetTextureCount(type) <= 0)
        return -1;
    if (amat->GetTexture(type, 0, &texname) != AI_SUCCESS)
        return -1;

    auto it = texture_index_map.find(ai_convert(texname));
    if (it != texture_index_map.end())
        return it->second;

    shared_ptr<Texture> texture = make_shared<Texture>();
    string filename = (folder / texname.C_Str()).string();
    texture->image.load_from_file(filename);
    int texture_id = scene->add_texture(texture);
    texture_index_map[ai_convert(texname)] = texture_id;

    return texture_id;
}

void AssimpImporter::traverse(aiNode* anode, shared_ptr<SceneGraphNode> node, const Transform& parent_transform)
{
    node->name = ai_convert(anode->mName);
    node->transform = ai_convert(anode->mTransformation);
    scene_graph_nodes[node->name] = node;

    Transform global_transform = parent_transform * node->transform;
    for (int i = 0; i < anode->mNumMeshes; i++)
    {
        // add instance
        int mesh_id = anode->mMeshes[i];
        int instance_id = scene->instances.size();
        scene->instances.push_back(mesh_id);
        scene->instance_transforms.push_back(global_transform);
        node->instance_ids.push_back(instance_id);
    }

    auto it = bone_index_map.find(node->name);
    if (it != bone_index_map.end())
        node->bone_id = it->second;

    for (int i = 0; i < anode->mNumChildren; i++)
    {
        shared_ptr<SceneGraphNode> child = make_shared<SceneGraphNode>();
        node->children.push_back(child);
        traverse(anode->mChildren[i], child, global_transform);
    }
}

void AssimpImporter::load_animations()
{
    for (int i = 0; i < ascene->mNumAnimations; i++)
    {
        aiAnimation* aanim = ascene->mAnimations[i];
        float duration = aanim->mDuration;
        float ticks_per_second = aanim->mTicksPerSecond;
        float duration_in_seconds = duration / ticks_per_second;

        for (int j = 0; j < aanim->mNumChannels; j++)
        {
            aiNodeAnim* achannel = aanim->mChannels[j];
            string name = ai_convert(achannel->mNodeName);
            auto it = scene_graph_nodes.find(name);
            if (it == scene_graph_nodes.end())
                continue;

            shared_ptr<SceneGraphNode> node = it->second;
            shared_ptr<Animation> animation = make_shared<Animation>();

            for (int k = 0; k < achannel->mNumPositionKeys; k++)
            {
                aiVectorKey key = achannel->mPositionKeys[k];
                float3 translation = ai_convert(key.mValue);
                float time_stamp = key.mTime / ticks_per_second;
                animation->translations.push_back({ time_stamp, translation });
            }
            for (int k = 0; k < achannel->mNumRotationKeys; k++)
            {
                aiQuatKey key = achannel->mRotationKeys[k];
                Quaternion rotation = ai_convert(key.mValue);
                float time_stamp = key.mTime / ticks_per_second;
                animation->rotations.push_back({ time_stamp, rotation });
            }
            for (int k = 0; k < achannel->mNumScalingKeys; k++)
            {
                aiVectorKey key = achannel->mScalingKeys[k];
                float3 scale = ai_convert(key.mValue);
                float time_stamp = key.mTime / ticks_per_second;
                animation->scales.push_back({ time_stamp, scale });
            }
            animation->duration = duration_in_seconds;

            int id = scene->add_animation(animation);
            node->animation_id = id;
        }
    }
}

shared_ptr<TriangleMesh> AssimpImporter::import_mesh(const string& filename)
{
    Assimp::Importer importer;
    unsigned int flags = aiProcess_Triangulate
        | aiProcess_GenNormals
        | aiProcess_CalcTangentSpace
        | aiProcess_FlipUVs;
    ascene = importer.ReadFile(filename, flags);

    if (!ascene || ascene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !ascene->mRootNode)
    {
        cout << "ERROR::ASSIMP::" << importer.GetErrorString() << endl;
        exit(-1);
    }

    assert(ascene->mNumMeshes > 0);
    aiMesh* amesh = ascene->mMeshes[0];

    vector<float3> vertices;
    vector<uint3> indices;
    vector<float3> normals;
    vector<float3> tangents;
    vector<float2> texcoords;
    for (int j = 0; j < amesh->mNumVertices; j++)
    {
        vertices.push_back(ai_convert(amesh->mVertices[j]));
        if (amesh->HasNormals())
            normals.push_back(ai_convert(amesh->mNormals[j]));
        if (amesh->HasTangentsAndBitangents())
            tangents.push_back(ai_convert(amesh->mTangents[j]));
        if (amesh->HasTextureCoords(0))
        {
            float3 texcoord = ai_convert(amesh->mTextureCoords[0][j]);
            texcoords.push_back({ texcoord.x, texcoord.y });
        }
    }
    for (int j = 0; j < amesh->mNumFaces; j++)
    {
        aiFace face = amesh->mFaces[j];
        assert(face.mNumIndices == 3);
        indices.push_back({ face.mIndices[0], face.mIndices[1], face.mIndices[2] });
    }
    string name = ai_convert(amesh->mName);

    return make_shared<TriangleMesh>(name, vertices, indices, normals, tangents, texcoords);
}

void AssimpImporter::import_scene(const string& filename, shared_ptr<Scene> _s)
{
    Assimp::Importer importer;
    unsigned int flags = aiProcess_Triangulate
        | aiProcess_GenNormals
        | aiProcess_CalcTangentSpace
        | aiProcess_FlipUVs;
    ascene = importer.ReadFile(filename, flags);

    if (!ascene || ascene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !ascene->mRootNode)
    {
        cout << "ERROR::ASSIMP::" << importer.GetErrorString() << endl;
        exit(-1);
    }

    folder = std::filesystem::path(filename).parent_path();
    scene = _s;
    if (scene->root == nullptr)
        scene->root = make_shared<SceneGraphNode>();

    load_meshes();
    traverse(ascene->mRootNode, scene->root, Transform());
    load_animations();
}