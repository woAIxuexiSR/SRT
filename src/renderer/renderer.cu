#include "renderer.h"

Renderer::Renderer(int _w, int _h, shared_ptr<Scene> _scene)
    : width(_w), height(_h), scene(_scene)
{
    film = make_shared<Film>(width, height);

    shared_ptr<RenderProcess> pt = make_shared<PathTracer>(_w, _h, _scene);
    // shared_ptr<AccumulateProcess> ap = make_shared<AccumulateProcess>(_w, _h);
    shared_ptr<Denoise> dn = make_shared<Denoise>(_w, _h);
    shared_ptr<ToneMapping> tm = make_shared<ToneMapping>(ToneMappingType::None, _w, _h);
    processes.push_back(pt);
    // processes.push_back(ap);
    processes.push_back(dn);
    processes.push_back(tm);
}

void Renderer::run()
{
    {
        PROFILE("Render");
        for (int i = 0; i < 32; i++)
            for (auto process : processes)
                process->render(film);
    }

    {
        PROFILE("Save");
        film->save("hhh.png");
    }
    // film->save("hhh.exr");

    // shared_ptr<GUI> gui = make_shared<GUI>(width, height, scene->camera);
    // while(!gui->should_close())
    // {
    //     gui->begin_frame();

    //     for(auto process : processes)
    //     {
    //         process->render(film);
    //         process->render_ui();
    //     }

    //     gui->write_texture(film->get_pixels());
    //     gui->end_frame();
    // }
    Profiler::print();
    Profiler::reset();
}