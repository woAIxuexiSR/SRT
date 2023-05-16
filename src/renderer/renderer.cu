#include "renderer.h"

ImageRenderer::ImageRenderer(int _w, int _h, shared_ptr<Scene> _scene, string _filename)
    : width(_w), height(_h), scene(_scene), filename(_filename)
{
    film = make_shared<Film>(width, height);
}

void ImageRenderer::load_processes_from_config(const json& config)
{
    for (auto& c : config)
    {
        string name = c.at("name");

        shared_ptr<RenderProcess> process = RenderProcessFactory::create_process(name, c["params"]);
        process->set_enable(c.at("enable"));
        process->resize(width, height);
        process->set_scene(scene);

        processes.push_back(process);
    }
}

void ImageRenderer::run()
{
    PROFILE("Render");

    for (auto process : processes)
        process->render(film);
    film->save(filename);

    Profiler::stop();
    Profiler::print();
    Profiler::reset();
}

InteractiveRenderer::InteractiveRenderer(int _w, int _h, shared_ptr<Scene> _scene)
    : width(_w), height(_h), scene(_scene)
{
    film = make_shared<Film>(width, height);
    gui = make_shared<GUI>(width, height, scene->camera);
}

void InteractiveRenderer::load_processes_from_config(const json& config)
{
    for (auto& c : config)
    {
        string name = c.at("name");

        shared_ptr<RenderProcess> process = RenderProcessFactory::create_process(name, c["params"]);
        process->set_enable(c.at("enable"));
        process->resize(width, height);
        process->set_scene(scene);

        processes.push_back(process);
    }
}

void InteractiveRenderer::run()
{
    while(!gui->should_close())
    {
        gui->begin_frame();

        scene->render_ui();
        for (auto process : processes)
        {
            process->render(film);
            process->render_ui();
        }

        gui->end_frame();
        gui->write_texture(film->get_pixels());
    }
}