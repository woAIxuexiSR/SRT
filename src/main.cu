#include <iostream>
#include "renderer.h"

int main(int argc, char** argv)
{
    std::filesystem::path config_path(__FILE__);
    config_path = config_path.parent_path().parent_path() / "config" / "default.json";
    std::ifstream f(config_path);
    json config = json::parse(f);
    f.close();

    int width = config.at("width");
    int height = config.at("height");

    // build scene
    json scene_config = config.at("scene");

    shared_ptr<Scene> scene = make_shared<Scene>();
    json camera_config = scene_config.at("camera");
    auto vec_to_f3 = [](const vector<float>& v) -> float3 {
        return { v[0], v[1], v[2] };
    };
    Transform transform = Transform::LookAt(
        vec_to_f3(camera_config.at("position")),
        vec_to_f3(camera_config.at("target")),
        vec_to_f3(camera_config.at("up"))
    );
    float aspect = (float)width / (float)height;
    float fov = camera_config.at("fov");
    shared_ptr<Camera> camera = make_shared<Camera>();
    camera->set_type(Camera::Type::Perspective);
    camera->set_aspect_fov(aspect, fov);
    // shared_ptr<Camera> camera = make_shared<Camera>(Camera::Mode::Environment, aspect, fov);
    // camera->set_thin_lens(5.0f, 0.1f);
    float radius = length(vec_to_f3(camera_config.at("target")) - vec_to_f3(camera_config.at("position")));
    camera->set_controller(transform, radius, CameraController::Type::Orbit);


    json model_config = scene_config.at("model");
    scene->set_camera(camera);
    scene->load_from_model(model_config.at("path").get<string>());
    scene->build_device_data();

    // render
    json render_config = config.at("render");

    string output = render_config.at("output");
    ImageRenderer renderer(width, height, scene, output);
    renderer.load_passes_from_config(render_config.at("processes"));
    // InteractiveRenderer renderer(width, height, scene);
    // renderer.load_processes_from_config(render_config.at("processes"));

    renderer.run();

    return 0;
}