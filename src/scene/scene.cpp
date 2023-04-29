#include "scene.h"

TriangleMesh::TriangleMesh(const string& type, const unordered_map<string, string>& params, SquareMatrix<4> transform)
{
    if (type == "plymesh")
    {
        auto it = params.find("string filename");
        if (it == params.end())
        {
            cout << "plymesh: missing filename" << endl;
            exit(-1);
        }
        string plyPath = params.at("folderpath") + "/" + dequote(it->second);
        create_from_ply(plyPath);
    }
    else if (type == "trianglemesh")
        create_from_triangles(params);

    Transform T(Transpose(transform));
    for (int i = 0; i < vertices.size(); i++)
        vertices[i] = T.apply_point(vertices[i]);
    // for(int i = 0; i < normals.size(); i++)
        // normals[i] = transform * normals[i];
}

void TriangleMesh::create_from_ply(const string& plyPath)
{
    Assimp::Importer importer;
    const aiScene* scene = importer.ReadFile(plyPath, aiProcess_FlipUVs);

    if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode)
    {
        cout << "ERROR::ASSIMP::" << importer.GetErrorString() << endl;
        exit(-1);
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

        if (mesh->mTextureCoords[0])
        {
            float2 texcoord = make_float2(mesh->mTextureCoords[0][i].x, mesh->mTextureCoords[0][i].y);
            texcoords.push_back(texcoord);
        }
    }
    for (int i = 0; i < mesh->mNumFaces; i++)
    {
        aiFace face = mesh->mFaces[i];
        int num_vertex = face.mNumIndices;
        if (num_vertex < 3)
            continue;
        else
        {
            for (int j = 1; j < num_vertex - 1; j++)
            {
                uint3 index = make_uint3(face.mIndices[0], face.mIndices[j], face.mIndices[j + 1]);
                indices.push_back(index);
            }
        }
    }
}

void TriangleMesh::create_from_triangles(const unordered_map<string, string>& params)
{
    auto it = params.find("point3 P");
    if (it == params.end())
    {
        cout << "trianglemesh: missing points" << endl;
        exit(1);
    }
    vector<float> P = parse_to_vector<float>(it->second);
    vertices.resize(P.size() / 3);
    memcpy(vertices.data(), P.data(), P.size() * sizeof(float));

    it = params.find("integer indices");
    if (it == params.end())
    {
        cout << "trianglemesh: missing indices" << endl;
        exit(1);
    }
    vector<int> I = parse_to_vector<int>(it->second);
    indices.resize(I.size() / 3);
    memcpy(indices.data(), I.data(), I.size() * sizeof(int));

    it = params.find("normal N");
    if (it != params.end())
    {
        vector<float> N = parse_to_vector<float>(it->second);
        normals.resize(N.size() / 3);
        memcpy(normals.data(), N.data(), N.size() * sizeof(float));
    }

    it = params.find("point2 uv");
    if (it != params.end())
    {
        vector<float> UV = parse_to_vector<float>(it->second);
        texcoords.resize(UV.size() / 2);
        memcpy(texcoords.data(), UV.data(), UV.size() * sizeof(float));
    }
}

void Texture::load_from_file(const string& filename)
{
    string suffix = filename.substr(filename.find_last_of(".") + 1);

    uint2 res;
    int compontents;
    unsigned char* image = stbi_load(filename.c_str(), (int*)&res.x, (int*)&res.y, &compontents, STBI_rgb_alpha);

    if (!image)
    {
        std::cout << "Failed to load texture file " << filename << std::endl;
        exit(-1);
    }

    pixels = (unsigned*)image;
    resolution = res;
}


void Scene::add_mesh(const string& type, const unordered_map<string, string>& params, int material_id, SquareMatrix<4> transform)
{
    shared_ptr<TriangleMesh> mesh = make_shared<TriangleMesh>(type, params, transform);
    mesh->material_id = material_id;
    if (mat_to_tex.find(material_id) != mat_to_tex.end())
        mesh->texture_id = mat_to_tex[material_id];
    meshes.push_back(mesh);
}

int Scene::add_texture(const string& name, const unordered_map<string, string>& params)
{
    shared_ptr<Texture> tex = make_shared<Texture>();
    int texture_id = textures.size();
    auto it = params.find("string filename");
    if (it == params.end())
    {
        cout << "texture: missing filename" << endl;
        exit(-1);
    }
    string texFile = params.at("folderpath") + "/" + dequote(it->second);
    tex->load_from_file(texFile);
    named_textures[name] = texture_id;
    textures.push_back(tex);
    return texture_id;
}

int Scene::add_named_material(const string& name, const unordered_map<string, string>& params)
{
    auto it = params.find("string type");
    if (it == params.end())
    {
        cout << "material: missing type" << endl;
        exit(-1);
    }
    int material_id = add_material(dequote(it->second), params);
    named_materials[name] = material_id;
    return material_id;
}

int Scene::add_material(const string& type, const unordered_map<string, string>& params)
{
    shared_ptr<Material> mat = make_shared<Material>();
    int material_id = materials.size();
    if (type == "diffuse")
    {
        mat->type = MaterialType::Diffuse;
        auto it = params.find("rgb reflectance");
        if (it != params.end())
        {
            vector<float> color = parse_to_vector<float>(it->second);
            mat->color = make_float3(color[0], color[1], color[2]);
        }

        it = params.find("texture reflectance");
        if (it != params.end())
        {
            string textureName = dequote(it->second);
            if (named_textures.find(textureName) == named_textures.end())
            {
                cout << "texture " << textureName << " not found" << endl;
                exit(-1);
            }
            mat_to_tex[material_id] = named_textures[textureName];
        }
    }
    else if (type == "coateddiffuse")
    {
        
    }

    materials.push_back(mat);
    return material_id;
}

int Scene::add_light_material(const string& type, const unordered_map<string, string>& params)
{
    // type == "diffuse"
    shared_ptr<Material> mat = make_shared<Material>();
    int material_id = materials.size();
    mat->type = MaterialType::Diffuse;
    auto it = params.find("rgb L");
    if (it != params.end())
    {
        vector<float> emission = parse_to_vector<float>(it->second);
        mat->emission = make_float3(emission[0], emission[1], emission[2]);
    }
    materials.push_back(mat);
    return material_id;
}


int Scene::get_material_id(const string& name)
{
    if (named_materials.find(name) == named_materials.end())
    {
        cout << "material " << name << " not found" << endl;
        exit(-1);
    }
    return named_materials[name];
}

void Scene::load_from_model(const string& filename)
{
    Assimp::Importer importer;
    const aiScene* scene = importer.ReadFile(filename, aiProcess_Triangulate | aiProcess_GenNormals | aiProcess_FlipUVs);

    if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode)
    {
        cout << "ERROR::ASSIMP::" << importer.GetErrorString() << endl;
        exit(-1);
    }

    std::filesystem::path folder = std::filesystem::path(filename).parent_path();

    for (int i = 0; i < scene->mNumMeshes; i++)
    {
        aiMesh* amesh = scene->mMeshes[i];

        shared_ptr<TriangleMesh> mesh = make_shared<TriangleMesh>();
        for (int j = 0; j < amesh->mNumVertices; j++)
        {
            float3 vertex = make_float3(amesh->mVertices[j].x, amesh->mVertices[j].y, amesh->mVertices[j].z);
            mesh->vertices.push_back(vertex);

            if (amesh->HasNormals())
            {
                float3 normal = make_float3(amesh->mNormals[j].x, amesh->mNormals[j].y, amesh->mNormals[j].z);
                mesh->normals.push_back(normal);
            }

            if (amesh->mTextureCoords[0])
            {
                float2 texcoord = make_float2(amesh->mTextureCoords[0][j].x, amesh->mTextureCoords[0][j].y);
                mesh->texcoords.push_back(texcoord);
            }
        }
        for (int j = 0; j < amesh->mNumFaces; j++)
        {
            aiFace face = amesh->mFaces[j];
            uint3 index = make_uint3(face.mIndices[0], face.mIndices[1], face.mIndices[2]);
            mesh->indices.push_back(index);
        }

        aiMaterial* amaterial = scene->mMaterials[amesh->mMaterialIndex];
        shared_ptr<Material> material = make_shared<Material>();

        aiColor3D emissive, diffuse, specular;
        amaterial->Get(AI_MATKEY_COLOR_EMISSIVE, emissive);
        amaterial->Get(AI_MATKEY_COLOR_DIFFUSE, diffuse);
        amaterial->Get(AI_MATKEY_COLOR_SPECULAR, specular);
        if (!emissive.IsBlack())
        {
            material->type = MaterialType::Diffuse;
            material->emission = make_float3(emissive.r, emissive.g, emissive.b);
            material->color = make_float3(diffuse.r, diffuse.g, diffuse.b);
        }
        else if(i == 5)
        {
            // material->type = MaterialType::Dielectric;
            // material->color = make_float3(2.0f);
            material->type = MaterialType::Disney;
            material->color = make_float3(0.099, 0.24, 0.134);
            material->params[1] = 0.5f;
            material->params[2] = 0.5f;
            material->params[3] = 0.5f;
            material->params[9] = 0.5f;
            material->params[10] = 1.0f;
            // material->params[11] = 0.5f;
            // material->color = make_float3(2.0f);
            // material->params[3] = 0.001f;
            // material->params[11] = 1.0f;
        }
        else
        {
            material->type = MaterialType::Diffuse;
            material->color = make_float3(diffuse.r, diffuse.g, diffuse.b);
            // material->type = MaterialType::Disney;
            // material->color = make_float3(diffuse.r, diffuse.g, diffuse.b);
            // material->params[0] = 0.8f;
            // material->params[2] = 0.3f;
            // material->params[8] = 1.0f;
        }

        int material_id = materials.size();
        materials.push_back(material);
        mesh->material_id = material_id;

        meshes.push_back(mesh);
    }
}