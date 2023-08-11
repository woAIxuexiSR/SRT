#include "scene.h"

/* build functions  */

int Scene::add_material(shared_ptr<Material> material)
{
    int id = materials.size();
    if (material->name == "")
        material->name = "material_" + std::to_string(id);
    materials.push_back(material);
    return id;
}

int Scene::find_material(const string& name)
{
    for (int i = 0; i < materials.size(); i++)
        if (materials[i]->name == name)
            return i;
    return -1;
}

int Scene::add_texture(shared_ptr<Texture> texture)
{
    int id = textures.size();
    if (texture->name == "")
        texture->name = "texture_" + std::to_string(id);
    textures.push_back(texture);
    return id;
}

int Scene::find_texture(const string& name)
{
    for (int i = 0; i < textures.size(); i++)
        if (textures[i]->name == name)
            return i;
    return -1;
}

int Scene::add_animation(shared_ptr<Animation> animation)
{
    int id = animations.size();
    if (animation->name == "")
        animation->name = "animation_" + std::to_string(id);
    animations.push_back(animation);
    return id;
}

int Scene::find_bone(const string& name)
{
    for (int i = 0; i < bones.size(); i++)
        if (bones[i].name == name)
            return i;
    return -1;
}

int Scene::add_bone(const Bone& bone)
{
    int id = bones.size();
    bones.push_back(bone);
    bone_transforms.push_back(Transform());
    return id;
}

void Scene::add_instance(const Transform& transform, shared_ptr<TriangleMesh> mesh)
{
    int id = meshes.size();
    if (mesh->name == "")
        mesh->name = "mesh_" + std::to_string(id);
    meshes.push_back(mesh);
    instances.push_back(id);
    instance_transforms.push_back(transform);
}

/* GPU build functions */

void Scene::build_gscene()
{
    build_gscene_textures();
    build_gscene_materials();
    build_gscene_meshes();
    build_gscene_instances();
    build_gscene_lights();
}

void Scene::build_gscene_textures()
{
    int texture_num = (int)textures.size();

    gscene.texture_arrays.resize(texture_num);
    gscene.texture_objects.resize(texture_num);

    for (int i = 0; i < texture_num; i++)
    {
        auto& image = textures[i]->image;
        cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<uchar4>();
        int width = image.resolution.x;
        int height = image.resolution.y;
        int pitch = width * sizeof(uchar4);
        if (image.format == Image::Format::Float)
        {
            channel_desc = cudaCreateChannelDesc<float4>();
            pitch = width * sizeof(float4);
        }

        cudaArray_t& pixel_array = gscene.texture_arrays[i];
        checkCudaErrors(cudaMallocArray(&pixel_array, &channel_desc, width, height));
        checkCudaErrors(cudaMemcpy2DToArray(pixel_array, 0, 0, image.get_pixels(), pitch, pitch, height, cudaMemcpyHostToDevice));

        cudaResourceDesc res_desc = {};
        res_desc.resType = cudaResourceTypeArray;
        res_desc.res.array.array = pixel_array;

        cudaTextureDesc tex_desc = {};
        tex_desc.addressMode[0] = cudaAddressModeWrap;
        tex_desc.addressMode[1] = cudaAddressModeWrap;
        tex_desc.filterMode = cudaFilterModeLinear;
        tex_desc.readMode = cudaReadModeNormalizedFloat;
        if (image.format == Image::Format::Float)
            tex_desc.readMode = cudaReadModeElementType;
        tex_desc.normalizedCoords = 1;
        tex_desc.maxAnisotropy = 1;
        tex_desc.maxMipmapLevelClamp = 99;
        tex_desc.minMipmapLevelClamp = 0;
        tex_desc.mipmapFilterMode = cudaFilterModePoint;
        tex_desc.borderColor[0] = 1.0f;
        tex_desc.sRGB = 1;
        if (image.format == Image::Format::Float)
            tex_desc.sRGB = 0;

        checkCudaErrors(cudaCreateTextureObject(&gscene.texture_objects[i], &res_desc, &tex_desc, nullptr));
    }
}

void Scene::build_gscene_materials()
{
    int material_num = (int)materials.size();

    vector<GMaterial> gmaterials(material_num);
    for (int i = 0; i < material_num; i++)
    {
        gmaterials[i].bxdf = materials[i]->bxdf;
        gmaterials[i].base_color = materials[i]->base_color;
        gmaterials[i].emission_color = materials[i]->emission_color;
        gmaterials[i].intensity = materials[i]->intensity;
        if (materials[i]->color_tex_id != -1)
            gmaterials[i].color_tex = gscene.texture_objects[materials[i]->color_tex_id];
        if (materials[i]->normal_tex_id != -1)
            gmaterials[i].normal_tex = gscene.texture_objects[materials[i]->normal_tex_id];
    }
    gscene.material_buffer.resize_and_copy_from_host(gmaterials);
}

void Scene::build_gscene_meshes()
{
    int mesh_num = (int)meshes.size();

    gscene.vertex_buffer.resize(mesh_num);
    gscene.index_buffer.resize(mesh_num);
    gscene.normal_buffer.resize(mesh_num);
    gscene.tangent_buffer.resize(mesh_num);
    gscene.texcoord_buffer.resize(mesh_num);

    gscene.original_vertex_buffer.resize(mesh_num);
    gscene.original_normal_buffer.resize(mesh_num);
    gscene.original_tangent_buffer.resize(mesh_num);
    gscene.bone_id_buffer.resize(mesh_num);
    gscene.bone_weight_buffer.resize(mesh_num);
    if (has_bone())
        gscene.bone_transform_buffer.resize_and_copy_from_host(bone_transforms);

    for (int i = 0; i < mesh_num; i++)
    {
        auto mesh = meshes[i];

        gscene.vertex_buffer[i].resize_and_copy_from_host(mesh->vertices);
        gscene.index_buffer[i].resize_and_copy_from_host(mesh->indices);
        if (!meshes[i]->normals.empty())
            gscene.normal_buffer[i].resize_and_copy_from_host(mesh->normals);
        if (!meshes[i]->tangents.empty())
            gscene.tangent_buffer[i].resize_and_copy_from_host(mesh->tangents);
        if (!meshes[i]->texcoords.empty())
            gscene.texcoord_buffer[i].resize_and_copy_from_host(mesh->texcoords);

        if (mesh->has_bone)
        {
            gscene.original_vertex_buffer[i].resize_and_copy_from_host(mesh->vertices);
            if (!meshes[i]->normals.empty())
                gscene.original_normal_buffer[i].resize_and_copy_from_host(mesh->normals);
            if (!meshes[i]->tangents.empty())
                gscene.original_tangent_buffer[i].resize_and_copy_from_host(mesh->tangents);
            gscene.bone_id_buffer[i].resize_and_copy_from_host(mesh->bone_ids);
            gscene.bone_weight_buffer[i].resize_and_copy_from_host(mesh->bone_weights);
        }
    }

    vector<GTriangleMesh> gmeshes(mesh_num);
    for (int i = 0; i < mesh_num; i++)
    {
        gmeshes[i].vertices = gscene.vertex_buffer[i].data();
        gmeshes[i].indices = gscene.index_buffer[i].data();
        gmeshes[i].normals = gscene.normal_buffer[i].data();
        gmeshes[i].tangents = gscene.tangent_buffer[i].data();
        gmeshes[i].texcoords = gscene.texcoord_buffer[i].data();

        gmeshes[i].material = gscene.material_buffer.data() + meshes[i]->material_id;
    }
    gscene.mesh_buffer.resize_and_copy_from_host(gmeshes);
}

void Scene::build_gscene_instances()
{
    gscene.instance_transform_buffer.resize_and_copy_from_host(instance_transforms);
}

void Scene::build_gscene_lights()
{
    gscene.instance_light_id.resize(instances.size(), -1);

    int light_num = 0;
    for (int i = 0; i < (int)instances.size(); i++)
    {
        auto material = materials[meshes[instances[i]]->material_id];
        if (material->intensity <= 0.0f) continue;

        gscene.instance_light_id[i] = light_num;
        light_num++;
    }

    gscene.light_area_buffer.resize(light_num);
    vector<AreaLight> area_lights(light_num);
    float weight_sum = 0.0f;
    for (int i = 0; i < (int)instances.size(); i++)
    {
        if (gscene.instance_light_id[i] == -1) continue;

        int light_id = gscene.instance_light_id[i];
        auto mesh = meshes[instances[i]];
        auto material = materials[meshes[instances[i]]->material_id];

        int face_num = (int)mesh->indices.size();
        float area = 0.0f;
        vector<float> areas(face_num);
        for (int j = 0; j < face_num; j++)
        {
            auto& index = mesh->indices[j];
            auto v0 = mesh->vertices[index.x], v1 = mesh->vertices[index.y], v2 = mesh->vertices[index.z];
            areas[j] = length(cross(v1 - v0, v2 - v0)) * 0.5f;
            area += areas[j];
        }
        weight_sum += area * material->intensity;
        gscene.light_area_buffer[light_id].resize_and_copy_from_host(areas);

        area_lights[light_id].mesh = gscene.mesh_buffer.data() + instances[i];
        area_lights[light_id].transform = gscene.instance_transform_buffer.data() + i;

        area_lights[light_id].face_num = face_num;
        area_lights[light_id].areas = gscene.light_area_buffer[light_id].data();
        area_lights[light_id].area_sum = area;
    }
    gscene.area_light_buffer.resize_and_copy_from_host(area_lights);

    EnvironmentLight env_light;
    if (environment_map_id == -1)
    {
        env_light.type = EnvironmentLight::Type::Constant;
        env_light.emission_color = background;
    }
    else
    {
        env_light.type = EnvironmentLight::Type::UVMap;
        env_light.texture = gscene.texture_objects[environment_map_id];
    }
    gscene.environment_light_buffer.resize_and_copy_from_host(&env_light, 1);

    Light light;
    light.num = light_num;
    light.lights = gscene.area_light_buffer.data();
    light.weight_sum = weight_sum;
    light.env_light = gscene.environment_light_buffer.data();
    gscene.light_buffer.resize_and_copy_from_host(&light, 1);
}


/* useful functions */

void Scene::compute_aabb()
{
    aabb = AABB();
    for (int i = 0; i < instances.size(); i++)
    {
        auto mesh = meshes[instances[i]];
        aabb.expand(instance_transforms[i].apply_aabb(mesh->aabb));
    }
}

void Scene::update(float t)
{
    if (has_animation() && root && enable_animation)
    {
        update_node(root, t, Transform());
        update_gscene();
    }
}

void Scene::update_node(shared_ptr<SceneGraphNode> node, float t, const Transform& parent_transform)
{
    Transform node_transform = node->transform;
    if (node->animation_id != -1)
        node_transform = animations[node->animation_id]->get_transform(t);
    Transform global_transform = parent_transform * node_transform;

    // update instance transforms
    for (auto id : node->instance_ids)
        instance_transforms[id] = global_transform;
    // update bone transforms
    if (node->bone_id != -1)
        bone_transforms[node->bone_id] = global_transform * bones[node->bone_id].offset;

    for (auto& child : node->children)
        update_node(child, t, global_transform);
}

void Scene::update_gscene()
{
    gscene.instance_transform_buffer.copy_from_host(instance_transforms);

    if (!has_bone())
        return;

    gscene.bone_transform_buffer.copy_from_host(bone_transforms);
    Transform* bone_transforms = gscene.bone_transform_buffer.data();
    for (int i = 0; i < (int)meshes.size(); i++)
    {
        if (!meshes[i]->has_bone)
            continue;

        int vertex_num = meshes[i]->vertices.size();
        int bone_num = bones.size();
        int* bone_ids = gscene.bone_id_buffer[i].data();
        float* bone_weights = gscene.bone_weight_buffer[i].data();

        float3* vertices = gscene.vertex_buffer[i].data();
        float3* normals = gscene.normal_buffer[i].data();
        float3* tangents = gscene.tangent_buffer[i].data();
        float3* original_vertices = gscene.original_vertex_buffer[i].data();
        float3* original_normals = gscene.original_normal_buffer[i].data();
        float3* original_tangents = gscene.original_tangent_buffer[i].data();

        tcnn::parallel_for_gpu(vertex_num, [=] __device__(int idx) {

            vertices[idx] = make_float3(0.0f);
            if (normals) normals[idx] = make_float3(0.0f);
            if (tangents) tangents[idx] = make_float3(0.0f);

            for (int j = 0; j < MAX_BONE_PER_VERTEX; j++)
            {
                int bone_id = bone_ids[idx * MAX_BONE_PER_VERTEX + j];
                float bone_weight = bone_weights[idx * MAX_BONE_PER_VERTEX + j];

                if (bone_id == -1)
                {
                    if (j == 0)  // no bone assigned
                    {
                        vertices[idx] = original_vertices[idx];
                        if (normals) normals[idx] = original_normals[idx];
                        if (tangents) tangents[idx] = original_tangents[idx];
                    }
                    break;
                }
                if (bone_id >= bone_num)
                    continue;

                Transform& t = bone_transforms[bone_id];
                vertices[idx] += t.apply_point(original_vertices[idx]) * bone_weight;
                if (normals) normals[idx] += t.apply_vector(original_normals[idx]) * bone_weight;
                if (tangents) tangents[idx] += t.apply_vector(original_tangents[idx]) * bone_weight;
            }

        });
    }
    checkCudaErrors(cudaDeviceSynchronize());
}

void Scene::render_ui()
{
    if (!ImGui::CollapsingHeader("Scene"))
        return;

    ImGui::Checkbox("Enable animation", &enable_animation);

    if (ImGui::TreeNode("Camera"))
    {
        CameraController& controller = camera->controller;
        bool changed = false;

        ImGui::Text("position: (%.1f, %.1f, %.1f), target: (%.1f, %.1f, %.1f)",
            controller.pos.x, controller.pos.y, controller.pos.z,
            controller.target.x, controller.target.y, controller.target.z);
        ImGui::Text("z: (%.1f, %.1f, %.1f), x: (%.1f, %.1f, %.1f), y: (%.1f, %.1f, %.1f)",
            controller.z.x, controller.z.y, controller.z.z,
            controller.x.x, controller.x.y, controller.x.z,
            controller.y.x, controller.y.y, controller.y.z);

        ImGui::PushItemWidth(ImGui::GetWindowWidth() * 0.21f);
        changed |= ImGui::DragFloat("theta", &controller.theta, 0.1f, 1.0f, 179.0f);
        ImGui::SameLine();
        changed |= ImGui::DragFloat("phi", &controller.phi, 0.1f);
        ImGui::SameLine();
        changed |= ImGui::DragFloat("fov", &camera->fov, 0.1f, 1.0f, 90.0f);
        ImGui::PopItemWidth();

        changed |= ImGui::Combo("camera type", (int*)&camera->type, "Perspective\0Orthographic\0ThinLens\0Environment\0\0");
        changed |= ImGui::Combo("camera controller type", (int*)&controller.type, "Orbit\0FPS\0\0");

        if (changed)
        {
            controller.reset();
            camera->reset();
            camera->set_moved(true);
        }

        ImGui::TreePop();
    }

    if (ImGui::TreeNode("Material"))
    {
        bool changed = false;
        for (int i = 0; i < (int)materials.size(); i++)
        {
            if (ImGui::TreeNode(materials[i]->name.c_str()))
            {
                changed |= ImGui::Combo("type##Material", (int*)&materials[i]->bxdf.type, "Diffuse\0DiffuseTransmission\0Dielectric\0Disney\0\0");
                changed |= ImGui::ColorEdit3("base color", &materials[i]->base_color.x);

                if (materials[i]->intensity > 0.0f)
                {
                    changed |= ImGui::ColorEdit3("emission", &materials[i]->emission_color.x);
                    changed |= ImGui::DragFloat("intensity", &materials[i]->intensity, 0.5f, 0.1f, 100.0f);
                }
                if (materials[i]->bxdf.type == BxDF::Type::Disney)
                {
                    if (ImGui::TreeNode("disney parameters"))
                    {
                        ImGui::PushItemWidth(ImGui::GetWindowWidth() * 0.15f);
                        changed |= ImGui::DragFloat("ior        ", &materials[i]->bxdf.ior, 0.01f, 0.0f, 2.0f);
                        ImGui::SameLine();
                        changed |= ImGui::DragFloat("metallic      ", &materials[i]->bxdf.metallic, 0.01f, 0.0f, 1.0f);
                        ImGui::SameLine();
                        changed |= ImGui::DragFloat("subsurface  ", &materials[i]->bxdf.subsurface, 0.01f, 0.0f, 1.0f);


                        changed |= ImGui::DragFloat("roughness  ", &materials[i]->bxdf.roughness, 0.01f, 0.001f, 1.0f);
                        ImGui::SameLine();
                        changed |= ImGui::DragFloat("specular      ", &materials[i]->bxdf.specular, 0.01f, 0.0f, 1.0f);
                        ImGui::SameLine();
                        changed |= ImGui::DragFloat("specularTint", &materials[i]->bxdf.specularTint, 0.01f, 0.0f, 1.0f);

                        changed |= ImGui::DragFloat("anisotropic", &materials[i]->bxdf.anisotropic, 0.01f, 0.0f, 1.0f);
                        ImGui::SameLine();
                        changed |= ImGui::DragFloat("sheen         ", &materials[i]->bxdf.sheen, 0.01f, 0.0f, 1.0f);
                        ImGui::SameLine();
                        changed |= ImGui::DragFloat("sheenTint   ", &materials[i]->bxdf.sheenTint, 0.01f, 0.0f, 1.0f);

                        changed |= ImGui::DragFloat("clearcoat  ", &materials[i]->bxdf.clearcoat, 0.01f, 0.0f, 1.0f);
                        ImGui::SameLine();
                        changed |= ImGui::DragFloat("clearcoatGloss", &materials[i]->bxdf.clearcoatGloss, 0.01f, 0.001f, 1.0f);
                        ImGui::SameLine();
                        changed |= ImGui::DragFloat("specTrans   ", &materials[i]->bxdf.specTrans, 0.01f, 0.0f, 1.0f);
                        ImGui::PopItemWidth();

                        ImGui::TreePop();
                    }
                }
                ImGui::TreePop();
            }
        }
        if (changed) build_gscene_materials();

        ImGui::TreePop();
    }

    if (ImGui::TreeNode("Animation"))
    {
        for (int i = 0; i < (int)animations.size();i++)
        {
            if (ImGui::TreeNode(animations[i]->name.c_str()))
            {
                ImGui::Text("Duration: %.2f", animations[i]->duration);
                ImGui::Combo("Interpolation type", (int*)&animations[i]->itype, "Linear\0Cubic\0\0");
                ImGui::Combo("Extrapolation type", (int*)&animations[i]->etype, "Clamp\0Repeat\0Mirror\0\0");

                ImGui::TreePop();
            }
        }
        ImGui::TreePop();
    }
}