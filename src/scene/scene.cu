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
    animations.push_back(animation);
    return id;
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
    int num_textures = (int)textures.size();

    gscene.texture_arrays.resize(num_textures);
    gscene.texture_objects.resize(num_textures);

    for (int i = 0; i < num_textures; i++)
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
    int num_materials = (int)materials.size();

    vector<GMaterial> gmaterials(num_materials);
    for (int i = 0; i < num_materials; i++)
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

    gscene.animated_vertex_buffer.resize(mesh_num);
    gscene.animated_normal_buffer.resize(mesh_num);
    gscene.animated_tangent_buffer.resize(mesh_num);
    gscene.bone_id_buffer.resize(mesh_num);
    gscene.bone_weight_buffer.resize(mesh_num);

    for (int i = 0; i < mesh_num; i++)
    {
        auto mesh = meshes[i];

        gscene.animated_vertex_buffer[i].resize_and_copy_from_host(mesh->vertices);
        gscene.index_buffer[i].resize_and_copy_from_host(mesh->indices);
        if (!meshes[i]->normals.empty())
            gscene.animated_normal_buffer[i].resize_and_copy_from_host(mesh->normals);
        if (!meshes[i]->tangents.empty())
            gscene.animated_tangent_buffer[i].resize_and_copy_from_host(mesh->tangents);
        if (!meshes[i]->texcoords.empty())
            gscene.texcoord_buffer[i].resize_and_copy_from_host(mesh->texcoords);

        if (mesh->has_bone)
        {
            gscene.vertex_buffer[i].resize_and_copy_from_host(mesh->vertices);
            if (!meshes[i]->normals.empty())
                gscene.normal_buffer[i].resize_and_copy_from_host(mesh->normals);
            if (!meshes[i]->tangents.empty())
                gscene.tangent_buffer[i].resize_and_copy_from_host(mesh->tangents);
            gscene.bone_id_buffer[i].resize_and_copy_from_host(mesh->bone_ids);
            gscene.bone_weight_buffer[i].resize_and_copy_from_host(mesh->bone_weights);
        }
    }

    vector<GTriangleMesh> gmeshes(mesh_num);
    for (int i = 0; i < mesh_num; i++)
    {
        gmeshes[i].vertices = gscene.animated_vertex_buffer[i].data();
        gmeshes[i].indices = gscene.index_buffer[i].data();
        gmeshes[i].normals = gscene.animated_normal_buffer[i].data();
        gmeshes[i].tangents = gscene.animated_tangent_buffer[i].data();
        gmeshes[i].texcoords = gscene.texcoord_buffer[i].data();

        gmeshes[i].material = gscene.material_buffer.data() + meshes[i]->material_id;
    }
    gscene.mesh_buffer.resize_and_copy_from_host(gmeshes);
}

void Scene::build_gscene_instances()
{
    int num_instances = (int)instances.size();

    gscene.instance_buffer.resize(num_instances);
    gscene.instance_meshid_buffer.resize(num_instances);
    gscene.instance_transform_buffer.resize_and_copy_from_host(instance_transforms);

    vector<GInstance> ginstances(num_instances);
    for (int i = 0; i < num_instances; i++)
    {
        ginstances[i].transform = gscene.instance_transform_buffer.data() + i;
        ginstances[i].mesh = gscene.mesh_buffer.data() + instances[i];
    }
    gscene.instance_buffer.resize_and_copy_from_host(ginstances);
}

void Scene::build_gscene_lights()
{
    gscene.instance_light_id.resize(instances.size(), -1);

    int num_light = 0;
    for (int i = 0; i < (int)instances.size(); i++)
    {
        auto material = materials[meshes[instances[i]]->material_id];
        if (material->intensity <= 0.0f) continue;

        gscene.instance_light_id[i] = num_light;
        num_light++;
    }

    gscene.light_area_buffer.resize(num_light);
    vector<AreaLight> area_lights(num_light);
    float weight_sum = 0.0f;
    for (int i = 0; i < (int)instances.size(); i++)
    {
        auto material = materials[meshes[instances[i]]->material_id];
        if (gscene.instance_light_id[i] == -1) continue;

        int light_id = gscene.instance_light_id[i];
        auto mesh = meshes[instances[i]];

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

        area_lights[light_id].face_num = face_num;
        area_lights[light_id].instance = gscene.instance_buffer.data() + i;
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

    Light light(num_light, gscene.area_light_buffer.data(), weight_sum, gscene.environment_light_buffer.data());
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
    update_node(root, t, Transform());
    update_gscene();
}

void Scene::update_node(shared_ptr<SceneGraphNode> node, float t, const Transform& parent_transform)
{
    Transform node_transform = node->transform;
    if (node->animation_id != -1)
        node_transform = animations[node->animation_id]->get_transform(t);
    Transform global_transform = parent_transform * node_transform;

    for (auto id : node->instance_ids)
        instance_transforms[id] = global_transform;

    // if (node->instance_id != -1)
    // {
    //     auto instance = instances[node->instance_id];
    //     instance_transforms[node->instance_id] = global_transform;
    // }
    // else if (node->bone_id != -1)
    // {
    //     auto& bone = bones[node->bone_id];
    //     bone_transforms[node->bone_id] = global_transform * bone.offset;
    // }

    for (auto& child : node->children)
        update_node(child, t, global_transform);
}

void Scene::update_gscene()
{
    if (!bones.empty())
        gscene.bone_transform_buffer.copy_from_host(bone_transforms);
    gscene.instance_transform_buffer.copy_from_host(instance_transforms);
}

void Scene::render_ui()
{

}