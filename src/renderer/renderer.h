#pragma once

#include <filesystem>

#include "definition.h"
#include "my_math.h"
#include "gui.h"
#include "film.h"
#include "scene.h"
#include "integrator.h"
#include "scene/camera.h"


class ImageRender
{
private:
    string image_path;

    shared_ptr<Film> film;
    shared_ptr<Camera> camera;
    shared_ptr<OptixRayTracer> ray_tracer;

public:
    ImageRender(RenderParams params, shared_ptr<Scene> scene, string _path);

    void render();
};


class InteractiveRender
{
private:
    shared_ptr<Film> film;
    shared_ptr<Camera> camera;
    shared_ptr<OptixRayTracer> ray_tracer;
    shared_ptr<InteractiveGui> gui;

public:
    InteractiveRender(RenderParams params, shared_ptr<Scene> scene);

    void render();
};


class VideoRender
{
private:
    string video_path;
    int fps;

    shared_ptr<Film> film;
    shared_ptr<Camera> camera;
    shared_ptr<OptixRayTracer> ray_tracer;

public:
    VideoRender(RenderParams params, shared_ptr<Scene> scene, string _path, int _fps = 30);

    void render();
};


class MaterialAdjustRender
{
private:
    shared_ptr<Film> film;
    shared_ptr<Camera> camera;
    shared_ptr<Scene> scene;
    shared_ptr<Material> mat;
    shared_ptr<MaterialAdjuster> ray_tracer;
    shared_ptr<MaterialAdjustGui> gui;

public:
    MaterialAdjustRender(int _w, int _h);

    void render();
};