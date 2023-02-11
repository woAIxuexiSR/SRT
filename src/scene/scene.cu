#include "scene.h"

namespace std
{
    inline bool operator<(const tinyobj::index_t& _a, const tinyobj::index_t& _b)
    {
        if (_a.vertex_index < _b.vertex_index) return true;
        if (_a.vertex_index > _b.vertex_index) return false;
        if (_a.normal_index < _b.normal_index) return true;
        if (_a.normal_index > _b.normal_index) return false;
        return _a.texcoord_index < _b.texcoord_index;
    }
}

int addVertex(TriangleMesh* mesh, tinyobj::attrib_t& attributes, tinyobj::index_t& idx, std::map<tinyobj::index_t, int>& knownVertices)
{
    if (knownVertices.find(idx) != knownVertices.end()) return knownVertices[idx];

    int newIdx = mesh->vertices.size();
    knownVertices[idx] = newIdx;

    const float3* vertex_array = (const float3*)attributes.vertices.data();
    const float3* normal_array = (const float3*)attributes.normals.data();
    const float2* texcoord_array = (const float2*)attributes.texcoords.data();

    mesh->vertices.push_back(vertex_array[idx.vertex_index]);
    if (idx.normal_index >= 0) mesh->normals.push_back(normal_array[idx.normal_index]);
    if (idx.texcoord_index >= 0) mesh->texcoords.push_back(texcoord_array[idx.texcoord_index]);

    return newIdx;
}

void Model::loadObj(const std::string& objPath)
{
    stbi_set_flip_vertically_on_load(true);

    std::filesystem::path modelDir = std::filesystem::path(objPath).parent_path();

    tinyobj::attrib_t attributes;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string warn, err;

    bool result = tinyobj::LoadObj(&attributes, &shapes, &materials, &warn, &err, objPath.c_str(), modelDir.c_str());

    if (!result)
    {
        std::cout << "Failed to load Obj file " << objPath << " " << err << std::endl;
        exit(-1);
    }

    if (materials.empty())
    {
        std::cout << "Failed to parse materials..." << std::endl;
        exit(-1);
    }

    std::cout << "Done loading obj file " << objPath << " - found " << shapes.size() << " shapes with " << materials.size() << " materials." << std::endl;

    std::map<std::string, int> knownTextures;
    for (int shapeID = 0; shapeID < (int)shapes.size(); ++shapeID)
    {
        tinyobj::shape_t& shape = shapes[shapeID];

        std::set<int> materialIDs;
        for (auto id : shape.mesh.material_ids)
            materialIDs.insert(id);

        std::map<tinyobj::index_t, int> knownVertices;
        for (int materialID : materialIDs)
        {
            TriangleMesh* mesh = new TriangleMesh;
            for (int faceID = 0; faceID < shape.mesh.material_ids.size(); ++faceID)
            {
                if (shape.mesh.material_ids[faceID] != materialID) continue;

                tinyobj::index_t idx0 = shape.mesh.indices[3 * faceID];
                tinyobj::index_t idx1 = shape.mesh.indices[3 * faceID + 1];
                tinyobj::index_t idx2 = shape.mesh.indices[3 * faceID + 2];

                uint3 idx = make_uint3(addVertex(mesh, attributes, idx0, knownVertices),
                                       addVertex(mesh, attributes, idx1, knownVertices),
                                       addVertex(mesh, attributes, idx2, knownVertices));
                mesh->indices.push_back(idx);
            }
            mesh->textureId = loadTexture(knownTextures, materials[materialID].diffuse_texname, modelDir);
            float3 emission = (const float3&)materials[materialID].emission;
            float3 diffuse = (const float3&)materials[materialID].diffuse;
            float3 specular = (const float3&)materials[materialID].specular;
            float3 transmittance = (const float3&)materials[materialID].transmittance;
            if (length(emission) > 0.0f)
            {
                mesh->mat.setType(MaterialType::Emissive);
                mesh->mat.setColor(emission);
            }
            else if (length(transmittance) > 0.0f || materials[materialID].illum == 6 || materials[materialID].illum == 7)
            {
                mesh->mat.setType(MaterialType::Glass);
                // mesh->mat.setColor(transmittance);
                mesh->mat.setColor(make_float3(1.0f));
                mesh->mat.ior = materials[materialID].ior;
            }
            else
            {
                mesh->mat.setType(MaterialType::Disney);
                if (length(specular) > 0.0f)
                {
                    mesh->mat.setColor(specular);
                    mesh->mat.metallic = 1.0f;
                    mesh->mat.roughness = 0.0f;
                }
                else mesh->mat.setColor(diffuse);
            }

            if (mesh->vertices.empty()) delete mesh;
            else meshes.push_back(mesh);
        }
    }

    std::cout << "create " << meshes.size() << " meshes." << std::endl;
}

int Model::loadTexture(std::map<std::string, int>& knownTextures, const std::string& textureName, const std::string& modelDir)
{
    if (textureName == "") return -1;
    if (knownTextures.find(textureName) != knownTextures.end()) return knownTextures[textureName];

    std::filesystem::path filePath(modelDir);
    std::string texPath = textureName;
    std::replace(texPath.begin(), texPath.end(), '\\', '/');
    filePath /= texPath;

    uint2 resolution;
    int nComponenet;
    unsigned char* image = stbi_load(filePath.c_str(), (int*)&resolution.x, (int*)&resolution.y, &nComponenet, STBI_rgb_alpha);

    if (!image)
    {
        std::cout << "Failed to load texture file " << textureName << std::endl;
        return -1;
    }

    int newIdx = textures.size();
    knownTextures[textureName] = newIdx;

    Texture* texture = new Texture;
    texture->pixels = (unsigned*)image;
    texture->resolution = resolution;

    textures.push_back(texture);

    return newIdx;
}

Model::Model(const std::string& objPath)
{
    loadObj(objPath);
}

Model::~Model()
{
    for (auto mesh : meshes) delete mesh;
    for (auto texture : textures) delete texture;
}