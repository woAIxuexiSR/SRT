#pragma once

#include "definition.h"
#include "my_math.h"
#include "scene.h"
#include "film.h"

class RenderProcess
{
protected:
    bool enable;
    int width, height;
    shared_ptr<Scene> scene;

public:
    RenderProcess(int _w, int _h, shared_ptr<Scene> _s = nullptr)
        : enable(true), width(_w), height(_h), scene(_s) {}

    virtual void render(shared_ptr<Film> film) = 0;
    virtual void render_ui() {};
};