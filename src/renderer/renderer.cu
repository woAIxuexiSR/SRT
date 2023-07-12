#include "renderer.h"

void Renderer::load_passes(const json& config)
{
    for (auto& c : config)
    {
        string name = c.at("name");

        shared_ptr<RenderPass> pass = RenderPassFactory::create_pass(name, c["params"]);
        pass->set_enable(c.value("enable", true));
        pass->resize(width, height);
        pass->set_scene(scene);

        passes.push_back(pass);
    }
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
        resize(parser.width, parser.height);
    }
    else
    {
        AssimpImporter importer;
        importer.import_scene(model_path, scene);
    }

    // load camera
    auto vec_to_f3 = [](const vector<float>& v) -> float3 { return { v[0], v[1], v[2] }; };
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

    scene->compute_aabb();
    scene->build_gscene();
    film = make_shared<Film>(width, height);
}

void ImageRenderer::run()
{
    {
        PROFILE("render");
        for (auto pass : passes)
            pass->render(film);
    }

    {
        PROFILE("save");
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