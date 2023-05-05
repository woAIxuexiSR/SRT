#include "mesh.h"

void TriangleMesh::compute_aabb()
{
    assert(!vertices.empty());
    aabb = AABB(vertices[0]);
    for (int i = 1; i < vertices.size(); i++)
        aabb.expand(vertices[i]);
}

void TriangleMesh::load_from_file(const string& filename)
{
    Assimp::Importer importer;
    const aiScene* scene = importer.ReadFile(filename, aiProcess_Triangulate | aiProcess_FlipUVs);

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

void TriangleMesh::load_from_triangles(const vector<float3>& _v, const vector<uint3>& _i, const vector<float3>& _n, const vector<float2>& _t)
{
    vertices = _v;
    indices = _i;
    normals = _n;
    texcoords = _t;

    compute_aabb();
}