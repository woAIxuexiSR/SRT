#include "scene.h"

void TriangleMesh::addVertices(const std::vector<float3>& _v, const std::vector<uint3>& _i)
{
    unsigned firstVertexID = vertices.size();
    vertices.insert(vertices.end(), _v.begin(), _v.end());

    uint3 offset = make_uint3(firstVertexID);
    for (const auto& i : _i)
        indices.push_back(i + offset);
}

void TriangleMesh::addCube(const float3& center, const float3& size)
{
    float3 leftBackDown = center - size / 2.0f;
    float3 rightFrontUp = center + size / 2.0f;

    unsigned firstVertexID = vertices.size();

    vertices.push_back(leftBackDown);
    vertices.push_back(make_float3(rightFrontUp.x, leftBackDown.y, leftBackDown.z));
    vertices.push_back(make_float3(leftBackDown.x, rightFrontUp.y, leftBackDown.z));
    vertices.push_back(make_float3(rightFrontUp.x, rightFrontUp.y, leftBackDown.z));
    vertices.push_back(make_float3(leftBackDown.x, leftBackDown.y, rightFrontUp.z));
    vertices.push_back(make_float3(rightFrontUp.x, leftBackDown.y, rightFrontUp.z));
    vertices.push_back(make_float3(leftBackDown.x, rightFrontUp.y, rightFrontUp.z));
    vertices.push_back(rightFrontUp);

    uint3 _i[] =
    {
        {0, 1, 3}, {2, 3, 0},
        {5, 7, 6}, {5, 6, 4},
        {0, 4, 5}, {0, 5, 1},
        {2, 3, 7}, {2, 7, 6},
        {1, 5, 7}, {1, 7, 3},
        {4, 0, 2}, {4, 2, 6}
    };

    uint3 offset = make_uint3(firstVertexID);
    for (const auto& i : _i)
        indices.push_back(i + offset);
}

void TriangleMesh::addSquare_XZ(const float3& center, const float2& size)
{
    unsigned firstVertexID = vertices.size();

    float3 half = make_float3(size.x / 2.0f, 0.0f, size.y / 2.0f);
    vertices.push_back(center - half);
    vertices.push_back(make_float3(center.x - half.x, center.y, center.z + half.z));
    vertices.push_back(make_float3(center.x + half.x, center.y, center.z - half.z));
    vertices.push_back(center + half);

    uint3 _i[] = { {0, 1, 3}, {2, 3, 0} };

    uint3 offset = make_uint3(firstVertexID);
    for (const auto& i : _i)
        indices.push_back(i + offset);
}

void TriangleMesh::addSquare_XY(const float3& center, const float2& size)
{
    unsigned firstVertexID = vertices.size();

    float3 half = make_float3(size.x / 2.0f, size.y / 2.0f, 0.0f);
    vertices.push_back(center - half);
    vertices.push_back(make_float3(center.x - half.x, center.y + half.y, center.z));
    vertices.push_back(make_float3(center.x + half.x, center.y - half.y, center.z));
    vertices.push_back(center + half);

    uint3 _i[] = { {0, 1, 3}, {2, 3, 0} };
    uint3 offset = make_uint3(firstVertexID);
    for (const auto& i : _i)
        indices.push_back(i + offset);
}

void TriangleMesh::addSquare_YZ(const float3& center, const float2& size)
{
    unsigned firstVertexID = vertices.size();

    float3 half = make_float3(0.0f, size.x / 2.0f, size.y / 2.0f);
    vertices.push_back(center - half);
    vertices.push_back(make_float3(center.x, center.y - half.y, center.z + half.z));
    vertices.push_back(make_float3(center.x, center.y + half.y, center.z - half.z));
    vertices.push_back(center + half);

    uint3 _i[] = { {0, 1, 3}, {2, 3, 0} };
    uint3 offset = make_uint3(firstVertexID);
    for (const auto& i : _i)
        indices.push_back(i + offset);
}

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
        exit(5);
    }

    if (materials.empty())
    {
        std::cout << "Failed to parse materials..." << std::endl;
        exit(5);
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
            DiffuseMaterial mat((const float3&)materials[materialID].diffuse);
            mesh->materialId = diffuseMaterials.size();
            diffuseMaterials.push_back(mat);
            mesh->textureId = loadTexture(knownTextures, materials[materialID].diffuse_texname, modelDir);
            mesh->emittance = (const float3&)materials[materialID].emission;

            if (mesh->vertices.empty()) delete mesh;
            else meshes.push_back(mesh);
        }
    }

    std::cout << "create " << meshes.size() << " meshes." << std::endl;
}

void Model::loadCornellBox()
{
    // const float3 white{ 0.8f, 0.8f, 0.8f };
    // const float3 red{ 0.8f, 0.05f, 0.05f };
    // const float3 green{ 0.05f, 0.8f, 0.05f };

    // TriangleMesh* box1 = new TriangleMesh;
    // box1->addCube(float3{ 186.0f, 82.5f, 169.5f }, float3{ 160.0f, 165.0f, 160.0f });
    // box1->setColor(white);
    // meshes.push_back(box1);

    // TriangleMesh* box2 = new TriangleMesh;
    // box2->addCube(float3{368.5f, 165.0f, 351.0f}, float3{158.0f, 330.0f, 159.0f});
    // box2->setColor(white);
    // meshes.push_back(box2);

    // TriangleMesh* ceiling = new TriangleMesh;
    // ceiling->addSquare_XZ(float3{278.0f, 548.8f, 279.6f}, float2{556.0f, 559.2f});
    // ceiling->setColor(white);
    // meshes.push_back(ceiling);

    // TriangleMesh* floor = new TriangleMesh;
    // floor->addSquare_XZ(float3{278.0f, 0.0f, 279.6f}, float2{556.0f, 559.2f});
    // floor->setColor(white);
    // meshes.push_back(floor);

    // TriangleMesh* leftwall = new TriangleMesh;
    // leftwall->addSquare_YZ(float3{556.0f, 274.4f, 279.6f}, float2{548.8f, 559.2f});
    // leftwall->setColor(red);
    // meshes.push_back(leftwall);

    // TriangleMesh* rightwall = new TriangleMesh;
    // rightwall->addSquare_YZ(float3{0.0f, 274.4f, 279.6f}, float2{548.8f, 559.2f});
    // rightwall->setColor(green);
    // meshes.push_back(rightwall);

    // TriangleMesh* backwall = new TriangleMesh;
    // backwall->addSquare_XY(float3{278.0f, 274.4f, 559.2f}, float2{556.0f, 548.8f});
    // backwall->setColor(white);
    // meshes.push_back(backwall);

    // TriangleMesh* light = new TriangleMesh;
    // light->addSquare_XZ(float3{278.0f, 548.6f, 279.6f}, float2{130.0f, 105.0f});
    // light->setEmittance(float3{15.0f});
    // meshes.push_back(light);

    /*
    TriangleMesh* occlude = new TriangleMesh;
    occlude->addCube(float3(278, 448.6, 279.6), float3(420, 80, 420));
    occlude->setColor(white);
    meshes.push_back(occlude);
    */
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