#include "renderer.h"

Renderer::Renderer(int _w, int _h, shared_ptr<Scene> _scene)
    : width(_w), height(_h), scene(_scene)
{
    film = make_shared<Film>(width, height);

    shared_ptr<RenderProcess> pt = make_shared<PathTracer>(_w, _h, _scene);
    shared_ptr<AccumulateProcess> ap = make_shared<AccumulateProcess>(_w, _h);
    processes.push_back(pt);
    processes.push_back(ap);
}

void Renderer::run()
{
    TICK(time);
    for (int t = 0; t < 32; t++)
    {
        for (auto process : processes)
            process->render(film);
    }
    TOCK(time);

    // film->save_ldr("hhh.png");
    film->save_ldr("hhh.jpg");
    // film->save_hdr("hhh.exr");
}