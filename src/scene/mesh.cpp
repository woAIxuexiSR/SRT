#include "mesh.h"

void TriangleMesh::compute_aabb()
{
    assert(!vertices.empty());
    aabb = AABB(vertices[0]);
    for (int i = 1; i < vertices.size(); i++)
        aabb.expand(vertices[i]);
}

void TriangleMesh::load_from_ply(const string& filename)
{
    std::ifstream f(filename, std::ios::binary);
    if (!f.is_open())
    {
        cout << "ERROR::Failed to open file: " << filename << endl;
        return;
    }

    tinyply::PlyFile file;
    file.parse_header(f);

    shared_ptr<tinyply::PlyData> vertices_data, faces_data, normals_data, texcoords_data;

    try { vertices_data = file.request_properties_from_element("vertex", { "x", "y", "z" }); }
    catch (const std::exception& e) { cout << "tinyply exception: " << e.what() << endl; }

    try { faces_data = file.request_properties_from_element("face", { "vertex_indices" }, 3); }
    catch (const std::exception& e) { cout << "tinyply exception: " << e.what() << endl; }

    try { normals_data = file.request_properties_from_element("vertex", { "nx", "ny", "nz" }); }
    catch (const std::exception& e) {}

    try { texcoords_data = file.request_properties_from_element("vertex", { "u", "v" }); }
    catch (const std::exception& e) {}

    file.read(f);

    if(vertices_data)
    {
        assert(vertices_data->t == tinyply::Type::FLOAT32);
        vertices.resize(vertices_data->count);
        memcpy(vertices.data(), vertices_data->buffer.get(), vertices_data->buffer.size_bytes());
    }
    if(faces_data)
    {
        assert(faces_data->t == tinyply::Type::UINT32 || faces_data->t == tinyply::Type::INT32);
        indices.resize(faces_data->count);
        memcpy(indices.data(), faces_data->buffer.get(), faces_data->buffer.size_bytes());
    }
    if(normals_data)
    {
        assert(normals_data->t == tinyply::Type::FLOAT32);
        normals.resize(normals_data->count);
        memcpy(normals.data(), normals_data->buffer.get(), normals_data->buffer.size_bytes());
    }
    if(texcoords_data)
    {
        assert(texcoords_data->t == tinyply::Type::FLOAT32);
        texcoords.resize(texcoords_data->count);
        memcpy(texcoords.data(), texcoords_data->buffer.get(), texcoords_data->buffer.size_bytes());
    }
    
    f.close();

    compute_aabb();
}

void TriangleMesh::load_from_others(const string& filename)
{
    Assimp::Importer importer;
    const aiScene* scene = importer.ReadFile(filename, aiProcess_ValidateDataStructure | aiProcess_Triangulate |
        aiProcess_GenSmoothNormals | aiProcess_PreTransformVertices);

    if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode)
    {
        cout << "ERROR::ASSIMP::" << importer.GetErrorString() << endl;
        return;
    }

    assert(scene->mNumMeshes == 1);

    aiMesh* mesh = scene->mMeshes[0];
    for (int i = 0; i < mesh->mNumVertices; i++)
    {
        float3 vertex = make_float3(mesh->mVertices[i].x, mesh->mVertices[i].y, mesh->mVertices[i].z);
        vertices.push_back(vertex);

        if (mesh->HasNormals())
        {
            float3 normal = make_float3(mesh->mNormals[i].x, mesh->mNormals[i].y, mesh->mNormals[i].z);
            normals.push_back(normal);
        }

        if (mesh->HasTextureCoords(0))
        {
            float2 texcoord = make_float2(mesh->mTextureCoords[0][i].x, mesh->mTextureCoords[0][i].y);
            texcoords.push_back(texcoord);
        }
    }
    for (int i = 0; i < mesh->mNumFaces; i++)
    {
        aiFace face = mesh->mFaces[i];
        assert(face.mNumIndices == 3);
        uint3 index = make_uint3(face.mIndices[0], face.mIndices[1], face.mIndices[2]);
        indices.push_back(index);
    }

    compute_aabb();
}

void TriangleMesh::load_from_file(const string& filename)
{
    string ext = filename.substr(filename.find_last_of(".") + 1);
    if (ext == "ply")
        load_from_ply(filename);
    else
        load_from_others(filename);
}

void TriangleMesh::load_from_triangles(const vector<float3>& _v, const vector<uint3>& _i, const vector<float3>& _n, const vector<float2>& _t)
{
    vertices = _v;
    indices = _i;
    normals = _n;
    texcoords = _t;

    compute_aabb();
}