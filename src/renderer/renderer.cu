#include "renderer.h"

Renderer::Renderer(const std::string& objPath, int _w, int _h, bool _u)
    : width(_w), height(_h), useGui(_u)
{
    film = std::make_shared<Film>(width, height);
    model = std::make_shared<Model>(objPath);
    camera = std::make_shared<Camera>(make_float3(0.0f, 1.0f, 0.5f), 5.0f, (float)width / (float)height);

    rayTracer = std::make_shared<PathTracer>(model.get(), width, height);

    if (useGui)
        gui = std::make_shared<Gui>(width, height, camera);
    else
    {
        std::filesystem::path curpath(__FILE__);
        outPath = curpath.parent_path().parent_path().parent_path() / "data" / "out.jpg";
    }
}

void Renderer::render()
{
    if (useGui)
    {
        while (!gui->shouldClose())
        {
            rayTracer->render(camera, film);
            film->fToUchar();
            gui->run((unsigned char*)film->getuPtr());
        }
    }
    else
    {
        rayTracer->render(camera, film);
        film->fToUchar();
        film->save_jpg(outPath.string());
        // film->save_exr(outPath.string());
    }
}