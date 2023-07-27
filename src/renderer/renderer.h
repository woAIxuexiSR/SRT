#pragma once

#include "definition.h"

#include "renderpass.h"
// #include "integrator/pathtracer.h"
// #include "npr/simple.h"
// #include "postprocess/accumulate.h"
// #include "postprocess/denoise.h"
// #include "postprocess/tonemapping.h"
// #include "wavefront/wavefront.h"

#include "gui.h"
#include "pbrtparse.h"
#include "assimp.h"


class Renderer
{
protected:
    std::filesystem::path config_path;

    int width, height;
    shared_ptr<Scene> scene;
    shared_ptr<Film> film;
    vector<shared_ptr<RenderPass> > passes;

public:
    Renderer(const string& _config_path) : config_path(_config_path) {}
    ~Renderer() { Profiler::reset(); }
    void set_scene(shared_ptr<Scene> _scene) { scene = _scene; }
    void resize(int _w, int _h) { width = _w, height = _h; }

    virtual void load_passes(const json& config);
    virtual void load_scene(const json& config);
    virtual void run() = 0;
};


class ImageRenderer : public Renderer
{
private:
    string filename;

public:
    ImageRenderer(const string& _config_path, const string& _filename)
        : Renderer(_config_path), filename(_filename) {}

    virtual void run() override;
};


class InteractiveRenderer : public Renderer
{
private:
    shared_ptr<GUI> gui;

public:
    InteractiveRenderer(const string& _config_path)
        : Renderer(_config_path) {}

    virtual void load_scene(const json& config) override;
    virtual void run() override;
};

class VideoRenderer : public Renderer
{
private:
    string filename;
    int frame;

public:
    VideoRenderer(const string& _config_path, const string& _filename, int _f)
        : Renderer(_config_path), filename(_filename), frame(_f) {}

    virtual void run() override;
};