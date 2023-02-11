#pragma once

#include <string>
#include <filesystem>
#include <memory>
#include <chrono>

#include "srt_math.h"
#include "gui.h"
#include "film.h"
#include "scene.h"
#include "scene/camera.h"
#include "optixRayTracer.h"
#include "pathTracer.h"
#include "bdpt.h"

class Renderer
{
private:
    int width, height;
    bool useGui;
    std::shared_ptr<Gui> gui;
    std::filesystem::path outPath;

    std::shared_ptr<Film> film;
    std::shared_ptr<Model> model;
    std::shared_ptr<Camera> camera;
    std::shared_ptr<OptixRayTracer> rayTracer;
    
    // Light light;
public:
    Renderer(const std::string& objPath, int _w = 800, int _h = 600, bool _u = true);

    void render();
};