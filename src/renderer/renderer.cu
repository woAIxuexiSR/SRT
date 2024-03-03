#include "renderer.h"

void Renderer::load_passes(const json& config)
{
    for (auto& c : config)
    {
        string name = c.at("name");

        shared_ptr<RenderPass> pass = RenderPassFactory::create_pass(name, c["params"]);
        pass->set_enable(c.value("enable", true));
        pass->set_online(online);
        pass->resize(width, height);
        pass->set_scene(scene);
        pass->init();

        passes.push_back(pass);
    }
}

void traverse_nodes(shared_ptr<SceneGraphNode> node, unordered_map<string, shared_ptr<SceneGraphNode>>& scene_graph_nodes)
{
    scene_graph_nodes[node->name] = node;
    for (auto child : node->children)
        traverse_nodes(child, scene_graph_nodes);
}

Camera::Type string_to_camera_type(const string& str)
{
    if (str == "perspective")
        return Camera::Type::Perspective;
    if (str == "orthographic")
        return Camera::Type::Orthographic;
    if (str == "thinlens")
        return Camera::Type::ThinLens;
    if (str == "environment")
        return Camera::Type::Environment;

    cout << "ERROR::Unknown camera type: " << str << endl;
    exit(-1);
}

void Renderer::load_scene(const json& config)
{
    // load model
    string model_path = config.at("model");
    model_path = (config_path.parent_path() / model_path).string();
    string ext = model_path.substr(model_path.find_last_of(".") + 1);

    scene = make_shared<Scene>();
    if (ext == "pbrt")
    {
        PBRTParser parser;
        parser.parse(model_path, scene);
        if (width == 0 || height == 0)
            resize(parser.width, parser.height);
        else
        {
            scene->camera->aspect = (float)width / (float)height;
            scene->camera->reset();
        }
    }
    else
    {
        AssimpImporter importer;
        importer.import_scene(model_path, scene);
    }
    if (width == 0 || height == 0)
        resize(1920, 1080);         // default resolution

    // load animation
    auto vec_to_f3 = [](const vector<float>& v) -> float3 { return { v[0], v[1], v[2] }; };
    auto vec_to_f4 = [](const vector<float>& v) -> float4 { return { v[0], v[1], v[2], v[3] }; };
    if (config.find("animation") != config.end())
    {
        json anim_config = config.at("animation");
        if (scene->root == nullptr)
        {
            scene->root = make_shared<SceneGraphNode>();
            scene->root->name = "root";
            scene->root->transform = Transform();
        }

        unordered_map<string, shared_ptr<SceneGraphNode>> scene_graph_nodes;
        traverse_nodes(scene->root, scene_graph_nodes);

        for (auto& c : anim_config)
        {
            string name = c.at("name");
            auto it = scene_graph_nodes.find(name);
            shared_ptr<SceneGraphNode> node;
            if (it == scene_graph_nodes.end())
            {
                node = make_shared<SceneGraphNode>();
                node->name = name;
                node->instance_ids.push_back(scene->find_instance(name));
                scene->root->children.push_back(node);
            }
            else node = it->second;

            if (node->animation_id != -1)
            {
                cout << "ERROR::Node " << name << " already has an animation" << endl;
                exit(-1);
            }

            shared_ptr<Animation> animation = make_shared<Animation>();
            animation->name = name;
            vector<float> time_stamps = c.at("keys");
            for (int i = 0; i < time_stamps.size(); i++)
            {
                if (c.find("translations") != c.end())
                    animation->translations.push_back({ time_stamps[i], vec_to_f3(c.at("translations")[i]) });
                if (c.find("rotations") != c.end())
                    animation->rotations.push_back({ time_stamps[i], vec_to_f4(c.at("rotations")[i]) });
                if (c.find("scales") != c.end())
                    animation->scales.push_back({ time_stamps[i], vec_to_f3(c.at("scales")[i]) });
            }
            animation->duration = time_stamps.back();
            int id = scene->add_animation(animation);
            node->animation_id = id;
        }
    }

    // load camera
    if (config.find("camera") != config.end())
    {
        json camera_config = config.at("camera");
        float3 position = vec_to_f3(camera_config.at("position"));
        float3 target = vec_to_f3(camera_config.at("target"));
        float3 up = vec_to_f3(camera_config.at("up"));
        float fov = camera_config.value("fov", 60.0f);
        float aspect = (float)width / (float)height;
        Camera::Type type = string_to_camera_type(camera_config.at("type"));

        shared_ptr<Camera> camera = make_shared<Camera>(type, aspect, fov);
        camera->set_controller(Transform::LookAt(position, target, up), length(position - target));

        camera->reset();
        scene->set_camera(camera);
    }
    else if (scene->camera == nullptr)
    {
        cout << "ERROR::No camera specified" << endl;
        exit(-1);
    }

    // load environment map
    if (config.find("environment") != config.end())
    {
        json env_config = config.at("environment");
        string env_type = env_config.at("type");
        if (env_type == "constant")
            scene->set_background(vec_to_f3(env_config.at("color")));
        else if (env_type == "uvmap")
        {
            string env_path = env_config.at("path");
            env_path = (config_path.parent_path() / env_path).string();
            shared_ptr<Texture> texture = make_shared<Texture>();
            texture->name = "environment";
            texture->image.load_from_file(env_path);
            scene->set_environment_map(texture);
        }
        else
        {
            cout << "ERROR::Unknown environment type: " << env_type << endl;
            exit(-1);
        }
    }

    scene->compute_mesh_area();
    scene->compute_aabb();
    scene->build_gscene();
    film = make_shared<Film>(width, height);
}

void ImageRenderer::run()
{
    {
        PROFILE("Render");
        for (auto pass : passes)
            pass->render(film);
    }

    {
        PROFILE("Save");
        Image image(width, height, film->get_pixels());
        image.save_to_file((config_path.parent_path() / filename).string());
    }

    Profiler::print();
}

void InteractiveRenderer::load_scene(const json& config)
{
    Renderer::load_scene(config);
    gui = make_shared<GUI>(width, height, scene->camera);
}

void InteractiveRenderer::run()
{
    float animt = 0.0f, t = 0.0f;
    CpuTimer timer;
    timer.start_timer();
    while (!gui->should_close())
    {
        gui->begin_frame();

        scene->update(t - animt);
        scene->render_ui();
        for (auto pass : passes)
        {
            pass->update();
            pass->render(film);
            pass->render_ui();
        }
        Profiler::render_ui();

        gui->write_texture(film->get_pixels());
        gui->end_frame();

        timer.end_timer();
        float delta = timer.get_time() * 0.001f;
        t += delta;
        if (!scene->enable_animation)
            animt += delta;
        timer.start_timer();
    }
}

void VideoRenderer::run()
{
    assert(frame > 0);

    string command = "mkdir temporary_images";
    int result = std::system(command.c_str());
    if (result != 0)
    {
        cout << "ERROR::Failed to create temporary_images folder" << endl;
        exit(-1);
    }

    float step = 1.0f / 30.0f;
    float t = 0.0f;
    for (int i = 0; i < frame; i++)
    {
        scene->update(t);
        for (auto pass : passes)
        {
            pass->update();
            pass->render(film);
        }
        Image image(width, height, film->get_pixels());
        image.save_to_file("temporary_images/" + std::to_string(i) + ".png");
        t += step;
    }

    command = "ffmpeg -y -framerate 30 -i temporary_images/%d.png ";
    command += "-c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p ";
    command += (config_path.parent_path() / filename).string();

    result = std::system(command.c_str());
    if (result != 0)
    {
        cout << "ERROR::Failed to create video" << endl;
        exit(-1);
    }

    command = "rm -rf temporary_images";
    result = std::system(command.c_str());
    if (result != 0)
    {
        cout << "ERROR::Failed to remove temporary_images folder" << endl;
        exit(-1);
    }
}