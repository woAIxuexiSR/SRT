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
    string output = config.value("output", "");

    // build scene
    json scene_config = config.at("scene");

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
    shared_ptr<Camera> camera = make_shared<Camera>(Camera::Mode::Perspective, aspect, fov);
    // shared_ptr<Camera> camera = make_shared<Camera>(Camera::Mode::Environment, aspect, fov);
    // camera->set_thin_lens(5.0f, 0.1f);
    camera->set_controller(CameraController::Type::Orbit, transform);

    // std::filesystem::path env_path(__FILE__);
    // env_path = env_path.parent_path().parent_path() / "data" / "skybox";
    // vector<string> faces = {
    //     // env_path / "right.jpg",
    //     // env_path / "left.jpg",
    //     // env_path / "top.jpg",
    //     // env_path / "bottom.jpg",
    //     // env_path / "front.jpg",
    //     // env_path / "back.jpg"
    //     env_path.parent_path() / "dam_wall_4k.hdr"
    // };

    json model_config = scene_config.at("model");
    shared_ptr<Scene> scene = make_shared<Scene>();
    scene->set_camera(camera);
    scene->load_from_model(model_config.at("path").get<string>());
    // scene->load_environment_map(faces);
    // scene->set_background({1.0f, 1.0f, 1.0f});

    Renderer renderer(width, height, scene);
    renderer.run();

    return 0;
}