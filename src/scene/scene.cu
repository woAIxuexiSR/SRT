#include "scene.h"

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

int Scene::add_texture(shared_ptr<Texture> texture, string name)
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
    for (int i = 0; i < num; i++)
    {
        shared_ptr<Texture> texture = make_shared<Texture>();
        texture->load_from_file(faces[i]);
        environment_map.push_back(texture);
    }
}

int Scene::get_material_id(const string& name) const
{
    for (int i = 0; i < material_names.size(); i++)
        if (material_names[i] == name)
            return i;
    return -1;
}

int Scene::get_texture_id(const string& name) const
{
    for (int i = 0; i < texture_names.size(); i++)
        if (texture_names[i] == name)
            return i;
    return -1;
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

    material->type = Material::Type::Disney;
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
            texture_id = get_texture_id(texname.C_Str());
            if (texture_id == -1)
            {
                shared_ptr<Texture> texture = make_shared<Texture>();
                texture->load_from_file((folder / texname.C_Str()).string());
                texture_id = add_texture(texture, texname.C_Str());
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

void Scene::build_device_data()
{
    create_device_meshes();
    create_device_materials();
    create_device_textures();
    create_device_environment_map();
    create_device_lights();
}

void Scene::create_device_meshes()
{
    int mesh_num = (int)meshes.size();

    d_scene.vertex_buffer.resize(mesh_num);
    d_scene.index_buffer.resize(mesh_num);
    d_scene.normal_buffer.resize(mesh_num);
    d_scene.texcoord_buffer.resize(mesh_num);

    for (int i = 0; i < mesh_num; i++)
    {
        d_scene.vertex_buffer[i].resize_and_copy_from_host(meshes[i]->vertices);
        d_scene.index_buffer[i].resize_and_copy_from_host(meshes[i]->indices);
        if (!meshes[i]->normals.empty())
            d_scene.normal_buffer[i].resize_and_copy_from_host(meshes[i]->normals);
        if (!meshes[i]->texcoords.empty())
            d_scene.texcoord_buffer[i].resize_and_copy_from_host(meshes[i]->texcoords);
    }
}

void Scene::create_device_materials()
{
    vector<Material> mats;
    for (auto& material : materials)
        mats.push_back(*material);
    d_scene.material_buffer.resize_and_copy_from_host(mats);
}

void Scene::create_device_textures()
{
    int num_textures = (int)textures.size();

    d_scene.texture_arrays.resize(num_textures);
    d_scene.texture_objects.resize(num_textures);

    for (int i = 0; i < num_textures; i++)
    {
        cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<uchar4>();
        int width = textures[i]->resolution.x;
        int height = textures[i]->resolution.y;
        int pitch = width * sizeof(uchar4);
        if (textures[i]->format == Texture::Format::Float)
        {
            channel_desc = cudaCreateChannelDesc<float4>();
            pitch = width * sizeof(float4);
        }

        cudaArray_t& pixel_array = d_scene.texture_arrays[i];
        checkCudaErrors(cudaMallocArray(&pixel_array, &channel_desc, width, height));
        checkCudaErrors(cudaMemcpy2DToArray(pixel_array, 0, 0, textures[i]->get_pixels(), pitch, pitch, height, cudaMemcpyHostToDevice));

        cudaResourceDesc res_desc = {};
        res_desc.resType = cudaResourceTypeArray;
        res_desc.res.array.array = pixel_array;

        cudaTextureDesc tex_desc = {};
        tex_desc.addressMode[0] = cudaAddressModeWrap;
        tex_desc.addressMode[1] = cudaAddressModeWrap;
        tex_desc.filterMode = cudaFilterModeLinear;
        tex_desc.readMode = cudaReadModeNormalizedFloat;
        if (textures[i]->format == Texture::Format::Float)
            tex_desc.readMode = cudaReadModeElementType;
        tex_desc.normalizedCoords = 1;
        tex_desc.maxAnisotropy = 1;
        tex_desc.maxMipmapLevelClamp = 99;
        tex_desc.minMipmapLevelClamp = 0;
        tex_desc.mipmapFilterMode = cudaFilterModePoint;
        tex_desc.borderColor[0] = 1.0f;
        tex_desc.sRGB = 0;

        checkCudaErrors(cudaCreateTextureObject(&d_scene.texture_objects[i], &res_desc, &tex_desc, nullptr));
    }
}

void Scene::create_device_environment_map()
{
    if (environment_map.empty())
        return;
    int num = (int)environment_map.size();
    assert(num == 1 || num == 6);

    Texture::Format format = environment_map[0]->format;

    cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<uchar4>();
    int width = environment_map[0]->resolution.x;
    int height = environment_map[0]->resolution.y;
    int pitch = width * sizeof(uchar4);
    if (format == Texture::Format::Float)
    {
        channel_desc = cudaCreateChannelDesc<float4>();
        pitch = width * sizeof(float4);
    }

    if (num == 1)
    {
        checkCudaErrors(cudaMallocArray(&d_scene.environment_map_array, &channel_desc, width, height));
        checkCudaErrors(cudaMemcpy2DToArray(d_scene.environment_map_array, 0, 0, environment_map[0]->get_pixels(), pitch, pitch, height, cudaMemcpyHostToDevice));
    }
    else
    {
        int channel = 6;
        cudaExtent extent = make_cudaExtent(width, height, channel);
        checkCudaErrors(cudaMalloc3DArray(&d_scene.environment_map_array, &channel_desc, extent, cudaArrayCubemap));

        vector<unsigned char> tex_data(pitch * height * channel);
        for (int i = 0; i < num; i++)
            memcpy(tex_data.data() + i * pitch * height, environment_map[i]->get_pixels(), pitch * height);

        cudaMemcpy3DParms copy_params = { 0 };
        copy_params.srcPtr = make_cudaPitchedPtr(tex_data.data(), pitch, width, height);
        copy_params.dstArray = d_scene.environment_map_array;
        copy_params.extent = extent;
        copy_params.kind = cudaMemcpyHostToDevice;
        checkCudaErrors(cudaMemcpy3D(&copy_params));
    }

    cudaResourceDesc res_desc = {};
    res_desc.resType = cudaResourceTypeArray;
    res_desc.res.array.array = d_scene.environment_map_array;

    cudaTextureDesc tex_desc = {};
    tex_desc.addressMode[0] = cudaAddressModeWrap;
    tex_desc.addressMode[1] = cudaAddressModeWrap;
    tex_desc.addressMode[2] = cudaAddressModeWrap;
    tex_desc.filterMode = cudaFilterModeLinear;
    tex_desc.readMode = cudaReadModeNormalizedFloat;
    if (format == Texture::Format::Float)
        tex_desc.readMode = cudaReadModeElementType;
    tex_desc.normalizedCoords = 1;
    tex_desc.maxAnisotropy = 1;
    tex_desc.maxMipmapLevelClamp = 99;
    tex_desc.minMipmapLevelClamp = 0;
    tex_desc.mipmapFilterMode = cudaFilterModePoint;
    tex_desc.borderColor[0] = 1.0f;
    tex_desc.sRGB = 0;

    checkCudaErrors(cudaCreateTextureObject(&d_scene.environment_map_object, &res_desc, &tex_desc, nullptr));
}

void Scene::create_device_lights()
{
    int mesh_num = (int)meshes.size();

    d_scene.meshid_to_lightid.resize(mesh_num);
    int cnt = 0;
    for (int i = 0; i < mesh_num; i++)
    {
        d_scene.meshid_to_lightid[i] = -1;
        if (!materials[meshes[i]->material_id]->is_emissive())
            continue;
        d_scene.meshid_to_lightid[i] = cnt;
        cnt++;
    }

    // build area lights
    vector<DiffuseAreaLight> area_lights(cnt);
    d_scene.light_area_buffer.resize(cnt);
    float weight_sum = 0.0f;
    for (int i = 0; i < mesh_num; i++)
    {
        if (d_scene.meshid_to_lightid[i] == -1)
            continue;

        int idx = d_scene.meshid_to_lightid[i];
        area_lights[idx].vertices = d_scene.vertex_buffer[i].data();
        area_lights[idx].indices = d_scene.index_buffer[i].data();
        area_lights[idx].normals = d_scene.normal_buffer[i].data();
        area_lights[idx].texcoords = d_scene.texcoord_buffer[i].data();
        if (meshes[i]->texture_id >= 0)
            area_lights[idx].texture = d_scene.texture_objects[meshes[i]->texture_id];

        area_lights[idx].emission_color = materials[meshes[i]->material_id]->emission_color;
        area_lights[idx].intensity = materials[meshes[i]->material_id]->intensity;

        int face_num = (int)meshes[i]->indices.size();
        vector<float> areas(face_num);
        float area_sum = 0.0f;
        for (int j = 0; j < face_num; j++)
        {
            uint3& index = meshes[i]->indices[j];
            float3 v0 = meshes[i]->vertices[index.x];
            float3 v1 = meshes[i]->vertices[index.y];
            float3 v2 = meshes[i]->vertices[index.z];
            areas[j] = 0.5f * length(cross(v1 - v0, v2 - v0));
            area_sum += areas[j];
        }
        d_scene.light_area_buffer[idx].resize_and_copy_from_host(areas);

        area_lights[idx].face_num = face_num;
        area_lights[idx].areas = d_scene.light_area_buffer[idx].data();
        area_lights[idx].area_sum = area_sum;

        weight_sum += area_sum * area_lights[idx].intensity;
    }
    d_scene.light_buffer.resize_and_copy_from_host(area_lights);

    // build environment light
    EnvironmentLight env_light;
    env_light.emission_color = background;
    if (environment_map.size() == 1)
    {
        env_light.type = EnvironmentLight::Type::UVMap;
        env_light.texture = d_scene.environment_map_object;
    }
    else if (environment_map.size() == 6)
    {
        env_light.type = EnvironmentLight::Type::CubeMap;
        env_light.texture = d_scene.environment_map_object;
    }
    d_scene.environment_light_buffer.resize_and_copy_from_host(&env_light, 1);

    // merge all lights
    d_scene.light.num = cnt;
    d_scene.light.lights = d_scene.light_buffer.data();
    d_scene.light.weight_sum = weight_sum;
    d_scene.light.environment = d_scene.environment_light_buffer.data();
}


void Scene::render_ui()
{
    if (!ImGui::CollapsingHeader("Scene"))
        return;

    CameraController& controller = camera->controller;
    if (ImGui::TreeNode("Camera"))
    {
        ImGui::DragFloat3("position", &controller.pos.x, 0.01f);
        ImGui::DragFloat3("target", &controller.target.x, 0.01f);

        ImGui::PushItemWidth(ImGui::GetWindowWidth() * 0.21f);
        bool theta = ImGui::DragFloat("theta", &controller.theta, 0.01f);
        ImGui::SameLine();
        bool phi = ImGui::DragFloat("phi", &controller.phi, 0.01f);
        ImGui::PopItemWidth();
        if (theta || phi) controller.reset_from_angle();

        ImGui::Text("z: (%.1f, %.1f, %.1f), x: (%.1f, %.1f, %.1f), y: (%.1f, %.1f, %.1f)",
            controller.z.x, controller.z.y, controller.z.z,
            controller.x.x, controller.x.y, controller.x.z,
            controller.y.x, controller.y.y, controller.y.z);

        ImGui::Combo("camera type", (int*)&camera->type, "Perspective\0Orthographic\0ThinLens\0Environment\0\0");
        if (camera->type == Camera::Type::ThinLens)
        {
            ImGui::PushItemWidth(ImGui::GetWindowWidth() * 0.21f);
            ImGui::DragFloat("focal", &camera->focal, 0.01f, 1.0f);
            ImGui::SameLine();
            ImGui::DragFloat("aperture", &camera->aperture, 0.01f, 0.0f);
            ImGui::PopItemWidth();
        }
        ImGui::Combo("camera controller type", (int*)&controller.type, "None\0Orbit\0FPS\0\0");

        ImGui::TreePop();
    }

    if (ImGui::TreeNode("Material"))
    {
        for (int i = 0; i < (int)materials.size(); i++)
        {
            if (ImGui::TreeNode(material_names[i].c_str()))
            {
                ImGui::Combo("type", (int*)&materials[i]->type, "Diffuse\0DiffuseTransmission\0Dielectric\0Disney\0\0");
                ImGui::ColorEdit3("color", &materials[i]->color.x);
                if (materials[i]->is_emissive())
                {
                    ImGui::ColorEdit3("emission color", &materials[i]->emission_color.x);
                    ImGui::DragFloat("intensity", &materials[i]->intensity, 0.01f);
                }

                if (materials[i]->type == Material::Type::DiffuseTransmission)
                    ImGui::ColorEdit3("transmission color", &materials[i]->params[0]);
                else if (materials[i]->type == Material::Type::Dielectric)
                    ImGui::DragFloat("ior", &materials[i]->params[0], 0.01f, 0.0f, 2.0f);
                else if (materials[i]->type == Material::Type::Disney)
                {
                    if (ImGui::TreeNode("disney parameters"))
                    {
                        ImGui::DragFloat("ior", &materials[i]->params[0], 0.01f, 0.0f, 2.0f);
                        ImGui::DragFloat("metallic", &materials[i]->params[1], 0.01f, 0.0f, 1.0f);
                        ImGui::DragFloat("subsurface", &materials[i]->params[2], 0.01f, 0.0f, 1.0f);
                        ImGui::DragFloat("roughness", &materials[i]->params[3], 0.01f, 0.0f, 1.0f);
                        ImGui::DragFloat("specular", &materials[i]->params[4], 0.01f, 0.0f, 1.0f);
                        ImGui::DragFloat("specular tint", &materials[i]->params[5], 0.01f, 0.0f, 1.0f);
                        // ImGui::DragFloat("anisotropic", &materials[i]->params[6], 0.01f, 0.0f, 1.0f);
                        ImGui::DragFloat("sheen", &materials[i]->params[7], 0.01f, 0.0f, 1.0f);
                        ImGui::DragFloat("sheen tint", &materials[i]->params[8], 0.01f, 0.0f, 1.0f);
                        ImGui::DragFloat("clearcoat", &materials[i]->params[9], 0.01f, 0.0f, 1.0f);
                        ImGui::DragFloat("clearcoat gloss", &materials[i]->params[10], 0.01f, 0.0f, 1.0f);
                        ImGui::DragFloat("specular transmission", &materials[i]->params[11], 0.01f, 0.0f, 1.0f);

                        ImGui::TreePop();
                    }
                }
                ImGui::TreePop();

                checkCudaErrors(cudaMemcpy(d_scene.material_buffer.data() + i, materials[i].get(), sizeof(Material), cudaMemcpyHostToDevice));
            }
        }
        ImGui::TreePop();
    }
}