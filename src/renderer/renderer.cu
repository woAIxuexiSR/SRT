#include "renderer.h"

void Renderer::resize(int _w, int _h)
{
    width = _w;
    height = _h;
}

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
    // load meshes
    json model_config = config.at("model");
    string type = model_config.at("type");
    string path = (config_path.parent_path() / model_config.at("path").get<string>()).string();

    // if (type == "pbrt")
    // {
    //     PBRTParser parser(path);
    //     parser.parse();
    //     set_scene(parser.scene);
    //     width = parser.width;
    //     height = parser.height;
    // }
    // else
    {
        scene = make_shared<Scene>();
        AssimpImporter importer;
        importer.import(path, scene);

        // load camera
        json camera_config = config.at("camera");
        shared_ptr<Camera> camera = make_shared<Camera>();
        auto vec_to_f3 = [](const vector<float>& v) -> float3 { return { v[0], v[1], v[2] }; };
        float3 position = vec_to_f3(camera_config.at("position"));
        float3 target = vec_to_f3(camera_config.at("target"));
        float3 up = vec_to_f3(camera_config.at("up"));

        camera->controller = CameraController(Transform::LookAt(position, target, up), length(position - target));
        camera->type = string_to_camera_type(camera_config.at("type"));
        camera->aspect = (float)width / (float)height;
        camera->fov = camera_config.value("fov", 60.0f);
        camera->fov /= camera->aspect;
        camera->focal = camera_config.value("focal", 1.0f);
        camera->aperture = camera_config.value("aperture", 0.0f);
        camera->reset();

        scene->set_camera(camera);

        // load environment light
        // if (config.find("environment") != config.end())
        // {
        //     json env_config = config.at("environment");
        //     string env_type = env_config.at("type");
        //     if (env_type == "constant")
        //         scene->set_background(vec_to_f3(env_config.at("color")));
        //     else if (env_type == "uvmap")
        //     {
        //         string env_path = (config_path.parent_path() / env_config.at("path").get<string>()).string();
        //         scene->load_environment_map(vector<string>({ env_path }));
        //     }
        //     else if (env_type == "cubemap")
        //     {
        //         vector<string> paths;
        //         for (auto& p : env_config.at("path"))
        //             paths.push_back((config_path.parent_path() / p.get<string>()).string());
        //         scene->load_environment_map(paths);
        //     }
        //     else
        //     {
        //         cout << "ERROR::Unknown environment type: " << env_type << endl;
        //         exit(-1);
        //     }
        // }
    }

    scene->compute_aabb();
    scene->build_gscene();
    film = make_shared<Film>(width, height);
}

void ImageRenderer::run()
{
    {
        PROFILE("render");
        float t = 0.2f;
        scene->update(t);
        for (auto pass : passes)
        {
            pass->update();
            pass->render(film);
        }

    }

    {
        PROFILE("save");
        film->save((config_path.parent_path() / filename).string());
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
    auto start = std::chrono::high_resolution_clock::now();
    float t = 0;
    while (!gui->should_close())
    {
        gui->begin_frame();

        // scene->update(t);
        scene->render_ui();
        for (auto pass : passes)
        {
            // pass->update();
            pass->render(film);
            pass->render_ui();
        }
        Profiler::render_ui();

        gui->write_texture(film->get_pixels());
        gui->end_frame();

        auto end = std::chrono::high_resolution_clock::now();
        t = std::chrono::duration<float>(end - start).count();
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

    auto start = std::chrono::high_resolution_clock::now();
    float t = 0;
    for (int i = 0; i < frame; i++)
    {
        scene->update(t);
        for (auto pass : passes)
        {
            pass->update();
            pass->render(film);
        }
        film->save("temporary_images/" + std::to_string(i) + ".png");
        auto end = std::chrono::high_resolution_clock::now();
        t = std::chrono::duration<float>(end - start).count() / 10;
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