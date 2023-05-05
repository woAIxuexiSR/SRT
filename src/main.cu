#include <iostream>
#include "renderer.h"

int main()
{
    const int width = 1920, height = 1080;

    std::filesystem::path current_path(__FILE__);
    std::filesystem::path model_path;
    model_path = current_path.parent_path().parent_path() / "data" / "CornellBox" / "CornellBox-Original.obj";

    Transform transform = LookAt({ 0.0f, 1.0f, 5.5f }, { 0.0f,1.0f,0.5f }, { 0.0f,1.0f,0.0f });
    float aspect = (float)width / (float)height;
    float fov = 60.0f;
    shared_ptr<Camera> camera = make_shared<Camera>(transform.get_matrix(), aspect, fov);

    shared_ptr<Scene> scene = make_shared<Scene>();
    scene->set_camera(camera);
    scene->load_from_model(model_path.string());

    Renderer renderer(width, height, scene);
    renderer.run();

    return 0;
}