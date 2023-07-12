#include <iostream>
#include "renderer.h"
#include "argparse.h"

json load_from_file(const string& path)
{
    std::ifstream f(path);
    if (!f.is_open())
    {
        cout << "ERROR::Failed to open file: " << path << endl;
        exit(-1);
    }

    json j = json::parse(f, nullptr, false, true);
    f.close();
    return j;
}

shared_ptr<Renderer> create_renderer(const string& path, const json& config)
{
    string type = config.at("type");
    string output = config.value("output", "");
    vector<int> resolution = config.value("resolution", vector<int>({ 1920, 1080 }));
    int frame = config.value("frame", 0);

    shared_ptr<Renderer> renderer;
    if (type == "image")
        renderer = make_shared<ImageRenderer>(path, output);
    else if (type == "interactive")
        renderer = make_shared<InteractiveRenderer>(path);
    else if (type == "video")
        renderer = make_shared<VideoRenderer>(path, output, frame);
    else
    {
        cout << "ERROR::Unknown renderer type: " << type << endl;
        exit(-1);
    }

    renderer->resize(resolution[0], resolution[1]);
    return renderer;
}

int main(int argc, char* argv[])
{
    std::filesystem::path path(__FILE__);
    string example_path = (path.parent_path().parent_path().parent_path() / "example" / "example.json").string();

    auto args = argparser("SRT Renderer")
        .set_program_name("main")
        .add_help_option()
        .add_option("-c", "--config", "Path to config file", example_path)
        .add_option("-r", "--render", "Override the render config", string(""))
        .add_option("-p", "--passes", "Override the passes config", string(""))
        .add_option("-s", "--scene", "Override the scene config", string(""))
        .parse(argc, argv);

    string config_path = args.get<string>("config");
    json config = load_from_file(config_path);
    if (args.get<string>("render") != "")
        config["render"] = load_from_file(args.get<string>("render"));
    if (args.get<string>("passes") != "")
        config["passes"] = load_from_file(args.get<string>("passes"));
    if (args.get<string>("scene") != "")
        config["scene"] = load_from_file(args.get<string>("scene"));

    auto renderer = create_renderer(config_path, config.at("render"));
    renderer->load_scene(config.at("scene"));
    renderer->load_passes(config.at("passes"));
    renderer->run();

    return 0;
}