#include "renderer.h"

Renderer::Renderer(int _w, int _h, shared_ptr<Scene> _scene)
    : width(_w), height(_h), scene(_scene)
{
    film = make_shared<Film>(width, height);

    shared_ptr<RenderProcess> pt = make_shared<PathTracer>(_w, _h, _scene);
    shared_ptr<AccumulateProcess> ap = make_shared<AccumulateProcess>(_w, _h);
    shared_ptr<DenoiseProcess> dn = make_shared<DenoiseProcess>(_w, _h);
    shared_ptr<ToneMappingProcess> tm = make_shared<ToneMappingProcess>(ToneMappingType::None, _w, _h);
    processes.push_back(pt);
    processes.push_back(ap);
    processes.push_back(dn);
    processes.push_back(tm);
}

void Renderer::run()
{
    TICK(time);
    for(int i = 0 ; i < 32; i++)
    for (auto process : processes)
        process->render(film);
    TOCK(time);

    // film->save_ldr("hhh.png");
    film->save_ldr("hhh2.jpg");
    // film->save_hdr("hhh.exr");

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
}