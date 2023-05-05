#pragma once

#include "definition.h"

#include "process.h"
#include "integrator/integrator.h"
#include "postprocess/accumulate.h"
#include "postprocess/tonemapping.h"
#include "postprocess/denoise.h"

#include "gui.h"

class Renderer
{
private:
    int width, height;
    shared_ptr<Scene> scene;

    shared_ptr<Film> film;
    vector<shared_ptr<RenderProcess> > processes;

public:
    Renderer(int _w, int _h, shared_ptr<Scene> _scene);

    void run();
};