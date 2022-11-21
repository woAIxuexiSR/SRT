#pragma once

#include "optixRayTracer.h"

class PathTracer : public OptixRayTracer
{
private:
    LaunchParams<int> launchParams;

public:
    PathTracer(const Model* _model, int _w, int _h);

    virtual void render(std::shared_ptr<Camera> camera, std::shared_ptr<Film> film) override;
};