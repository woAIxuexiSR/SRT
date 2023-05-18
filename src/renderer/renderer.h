#pragma once

#include "definition.h"

#include "process.h"
#include "integrator/integrator.h"
#include "postprocess/accumulate.h"
#include "postprocess/tonemapping.h"
#include "postprocess/denoise.h"

#include "gui.h"
#include "pbrtparse.h"

class ImageRenderer
{
private:
    int width, height;
    shared_ptr<Scene> scene;

    shared_ptr<Film> film;
    vector<shared_ptr<RenderProcess> > processes;

    string filename;

public:
    ImageRenderer(int _w, int _h, shared_ptr<Scene> _scene, string _filename);
    void load_processes_from_config(const json &config);
    void run();
};

class InteractiveRenderer
{
private:
    int width, height;
    shared_ptr<Scene> scene;

    shared_ptr<Film> film;
    vector<shared_ptr<RenderProcess> > processes;

    shared_ptr<GUI> gui;

public:
    InteractiveRenderer(int _w, int _h, shared_ptr<Scene> _scene);
    void load_processes_from_config(const json &config);
    void run();
};

class VideoRenderer
{
};