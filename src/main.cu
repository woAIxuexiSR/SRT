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
    scene->load_from_config(scene_config, width, height);

    // render
    json render_config = config.at("render");

    string output = render_config.at("output");
    ImageRenderer renderer(width, height, scene, output);
    renderer.load_processes_from_config(render_config.at("processes"));
    // InteractiveRenderer renderer(width, height, scene);
    // renderer.load_processes_from_config(render_config.at("processes"));

    renderer.run();

    return 0;
}