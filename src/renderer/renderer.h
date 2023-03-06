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

class Render
{
protected:
    std::shared_ptr<Film> film;
    std::shared_ptr<Model> model;
    std::shared_ptr<Camera> camera;
    std::shared_ptr<OptixRayTracer> rayTracer;

public:
    Render(
        std::shared_ptr<Film> _film,
        std::shared_ptr<Model> _model,
        std::shared_ptr<Camera> _camera,
        std::shared_ptr<OptixRayTracer> _rayTracer
    ): film(_film), model(_model), camera(_camera), rayTracer(_rayTracer) {}

    virtual void render() = 0;
};


class ImageRender: Render
{
private:
    std::string imagePath;
    std::string type;

public:
    ImageRender(
        std::shared_ptr<Film> _film,
        std::shared_ptr<Model> _model,
        std::shared_ptr<Camera> _camera,
        std::shared_ptr<OptixRayTracer> _rayTracer,
        std::string _imagePath,
        std::string _type = "EXR"
    ): Render(_film, _model, _camera, _rayTracer), imagePath(_imagePath), type(_type) {}


    virtual void render() override;
};


class InteractiveRender: Render
{
private:
    std::shared_ptr<Gui> gui;

public:
    InteractiveRender(
        std::shared_ptr<Film> _film,
        std::shared_ptr<Model> _model,
        std::shared_ptr<Camera> _camera,
        std::shared_ptr<OptixRayTracer> _rayTracer,
        std::shared_ptr<Gui> _gui
    ): Render(_film, _model, _camera, _rayTracer), gui(_gui) {}

    virtual void render() override;
};


class VideoRender: Render
{
private:
    std::string videoPath;
    int fps;

public:
    VideoRender(
        std::shared_ptr<Film> _film,
        std::shared_ptr<Model> _model,
        std::shared_ptr<Camera> _camera,
        std::shared_ptr<OptixRayTracer> _rayTracer,
        std::string _videoPath,
        int _fps = 30
    ): Render(_film, _model, _camera, _rayTracer), videoPath(_videoPath), fps(_fps) {}

    virtual void render() override;
};