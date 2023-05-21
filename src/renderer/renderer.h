#pragma once

#include "definition.h"

#include "renderpass.h"
#include "integrator/integrator.h"
#include "postprocess/accumulate.h"
#include "postprocess/tonemapping.h"
#include "postprocess/denoise.h"

#include "gui.h"
#include "pbrtparse.h"

class ImageRenderer
{
private:
    int width, height;
    shared_ptr<Scene> scene;

    shared_ptr<Film> film;
    vector<shared_ptr<RenderPass> > passes;

    string filename;

public:
    ImageRenderer(int _w, int _h, shared_ptr<Scene> _scene, string _filename);
    void load_passes_from_config(const json &config);
    void run();
};

class InteractiveRenderer
{
private:
    int width, height;
    shared_ptr<Scene> scene;

    shared_ptr<Film> film;
    vector<shared_ptr<RenderPass> > passes;

    shared_ptr<GUI> gui;

public:
    InteractiveRenderer(int _w, int _h, shared_ptr<Scene> _scene);
    void load_passes_from_config(const json &config);
    void run();
};

class VideoRenderer
{
};


// void Scene::load_from_config(const json& config, int width, int height)
// {
//     json camera_config = config.at("camera");
//     auto vec_to_f3 = [](const vector<float>& v) -> float3 {
//         return { v[0], v[1], v[2] };
//     };
//     Transform transform = Transform::LookAt(
//         vec_to_f3(camera_config.at("position")),
//         vec_to_f3(camera_config.at("target")),
//         vec_to_f3(camera_config.at("up"))
//     );
//     float aspect = (float)width / (float)height;
//     float fov = camera_config.at("fov");
//     shared_ptr<Camera> camera = make_shared<Camera>(Camera::Mode::Perspective, aspect, fov);
//     // shared_ptr<Camera> camera = make_shared<Camera>(Camera::Mode::Environment, aspect, fov);
//     // camera->set_thin_lens(5.0f, 0.1f);
//     float radius = length(vec_to_f3(camera_config.at("target")) - vec_to_f3(camera_config.at("position")));
//     camera->set_controller(CameraController::Type::Orbit, transform, radius);


//     json model_config = config.at("model");
//     set_camera(camera);
//     load_from_model(model_config.at("path").get<string>());
// }