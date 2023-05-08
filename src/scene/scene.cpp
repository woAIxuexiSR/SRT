#include "scene.h"

void set_material_property(shared_ptr<Material> material, const string& name, float value)
{
    if (name == "ior")
        material->params[0] = value;
    else if (name == "metallic")
        material->params[1] = value;
    else if (name == "subsurface")
        material->params[2] = value;
    else if (name == "roughness")
        material->params[3] = value;
    else if (name == "specular")
        material->params[4] = value;
    else if (name == "specularTint")
        material->params[5] = value;
    else if (name == "anisotropic")
        material->params[6] = value;
    else if (name == "sheen")
        material->params[7] = value;
    else if (name == "sheenTint")
        material->params[8] = value;
    else if (name == "clearcoat")
        material->params[9] = value;
    else if (name == "clearcoatGloss")
        material->params[10] = value;
    else if (name == "specTrans")
        material->params[11] = value;
    else
    {
        cout << "ERROR::UNKNOWN_MATERIAL_PROPERTY::" << name << endl;
        return;
    }
}

int Scene::add_mesh(shared_ptr<TriangleMesh> mesh, int material_id)
{
    mesh->material_id = material_id;
    if (material_to_texture.find(material_id) != material_to_texture.end())
        mesh->texture_id = material_to_texture[material_id];

    if (meshes.empty() && animated_meshes.empty())
        aabb = mesh->aabb;
    else
        aabb.expand(mesh->aabb);

    int id = meshes.size();
    meshes.push_back(mesh);
    return id;
}

int Scene::add_animated_mesh(shared_ptr<AnimatedTriangleMesh> mesh, int material_id)
{
    mesh->material_id = material_id;
    if (material_to_texture.find(material_id) != material_to_texture.end())
        mesh->texture_id = material_to_texture[material_id];

    if (meshes.empty() && animated_meshes.empty())
        aabb = mesh->aabb;
    else
        aabb.expand(mesh->aabb);

    int id = animated_meshes.size();
    animated_meshes.push_back(mesh);
    return id;
}

int Scene::add_material(shared_ptr<Material> material, string name, int texture_id)
{
    int id = materials.size();
    name = (name == "") ? "material_" + std::to_string(id) : name;
    materials.push_back(material);
    material_names.push_back(name);
    if (texture_id != -1)
        material_to_texture[id] = texture_id;
    return id;
}

int Scene::add_textures(shared_ptr<Texture> texture, string name)
{
    int id = textures.size();
    name = (name == "") ? "texture_" + std::to_string(id) : name;
    textures.push_back(texture);
    texture_names.push_back(name);
    return id;
}

void Scene::load_environment_map(const vector<string>& faces)
{
    int num = faces.size();
    assert(num == 1 || num == 6);
    for(int i = 0; i < num; i++)
    {
        shared_ptr<Texture> texture = make_shared<Texture>();
        texture->load_from_file(faces[i]);
        environment_map.push_back(texture);
    }
}

void convert_material(aiMaterial* amat, shared_ptr<Material> material)
{
    aiColor3D diffuse{ 0,0,0 };
    aiColor3D emission{ 0,0,0 };
    float emissiveStrength = 1.0f;
    float ior = 1.5f;
    float metallic = 0.0f;
    // float subsurface = 0.0f;
    float roughness = 0.5f;
    float specular = 0.5f;
    //float specularTint = 0.0f;
    //float anisotropic = 0.0f;
    float sheen = 0.0f;
    float sheenTint = 0.0f;
    float clearcoat = 0.0f;
    float clearcoatGloss = 0.0f;
    float specTrans = 0.0f;

    amat->Get(AI_MATKEY_COLOR_DIFFUSE, diffuse);
    amat->Get(AI_MATKEY_COLOR_EMISSIVE, emission);
    amat->Get(AI_MATKEY_EMISSIVE_INTENSITY, emissiveStrength);
    amat->Get(AI_MATKEY_REFRACTI, ior);
    amat->Get(AI_MATKEY_METALLIC_FACTOR, metallic);
    amat->Get(AI_MATKEY_ROUGHNESS_FACTOR, roughness);
    amat->Get(AI_MATKEY_SPECULAR_FACTOR, specular);
    amat->Get(AI_MATKEY_SHEEN_COLOR_FACTOR, sheen);
    amat->Get(AI_MATKEY_SHEEN_ROUGHNESS_FACTOR, sheenTint);
    amat->Get(AI_MATKEY_CLEARCOAT_FACTOR, clearcoat);
    amat->Get(AI_MATKEY_CLEARCOAT_ROUGHNESS_FACTOR, clearcoatGloss);
    amat->Get(AI_MATKEY_TRANSMISSION_FACTOR, specTrans);

    material->type = MaterialType::Disney;
    material->color = make_float3(diffuse.r, diffuse.g, diffuse.b);
    material->emission_color = make_float3(emission.r, emission.g, emission.b);
    material->intensity = emissiveStrength;
    set_material_property(material, "ior", ior);
    set_material_property(material, "metallic", metallic);
    set_material_property(material, "roughness", roughness);
    set_material_property(material, "specular", specular);
    set_material_property(material, "sheen", sheen);
    set_material_property(material, "sheenTint", sheenTint);
    set_material_property(material, "clearcoat", clearcoat);
    set_material_property(material, "clearcoatGloss", clearcoatGloss);
    set_material_property(material, "specTrans", specTrans);
}

void Scene::load_from_model(const string& filename)
{
    Assimp::Importer importer;
    const aiScene* scene = importer.ReadFile(filename, aiProcess_Triangulate | aiProcess_FlipUVs);

    if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode)
    {
        cout << "ERROR::ASSIMP::" << importer.GetErrorString() << endl;
        return;
    }

    std::filesystem::path folder = std::filesystem::path(filename).parent_path();
    for (int i = 0; i < scene->mNumMeshes; i++)
    {
        aiMesh* amesh = scene->mMeshes[i];

        vector<float3> vertices, normals;
        vector<uint3> indices;
        vector<float2> texcoords;
        for (int j = 0; j < amesh->mNumVertices; j++)
        {
            float3 vertex = make_float3(amesh->mVertices[j].x, amesh->mVertices[j].y, amesh->mVertices[j].z);
            vertices.push_back(vertex);

            if (amesh->HasNormals())
            {
                float3 normal = make_float3(amesh->mNormals[j].x, amesh->mNormals[j].y, amesh->mNormals[j].z);
                normals.push_back(normal);
            }

            if (amesh->HasTextureCoords(0))
            {
                float2 texcoord = make_float2(amesh->mTextureCoords[0][j].x, amesh->mTextureCoords[0][j].y);
                texcoords.push_back(texcoord);
            }
        }
        for (int j = 0; j < amesh->mNumFaces; j++)
        {
            aiFace& face = amesh->mFaces[j];
            uint3 index = make_uint3(face.mIndices[0], face.mIndices[1], face.mIndices[2]);
            indices.push_back(index);
        }

        // load mesh
        shared_ptr<TriangleMesh> mesh = make_shared<TriangleMesh>();
        mesh->load_from_triangles(vertices, indices, normals, texcoords);

        // load material
        aiMaterial* amat = scene->mMaterials[amesh->mMaterialIndex];
        shared_ptr<Material> material = make_shared<Material>();
        convert_material(amat, material);

        // load texture
        int texture_id = -1;
        aiString texname;
        if (amat->GetTextureCount(aiTextureType_DIFFUSE) > 0
            && amat->GetTexture(aiTextureType_DIFFUSE, 0, &texname) == AI_SUCCESS)
        {
            for (int j = 0; j < texture_names.size(); j++)
                if (std::strcmp(texture_names[j].c_str(), texname.C_Str()) == 0)
                {
                    texture_id = j;
                    break;
                }
            if (texture_id == -1)
            {
                shared_ptr<Texture> texture = make_shared<Texture>();
                texture->load_from_file(folder / texname.C_Str());
                texture_id = add_textures(texture, texname.C_Str());
            }
        }

        // add all
        aiString matname;
        int material_id = -1;
        if (amat->Get(AI_MATKEY_NAME, matname) == AI_SUCCESS)
            material_id = add_material(material, matname.C_Str(), texture_id);
        else
            material_id = add_material(material, "", texture_id);
        add_mesh(mesh, material_id);
    }
}

void Scene::render_ui()
{
}