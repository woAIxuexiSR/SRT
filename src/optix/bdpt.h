#pragma once

#include "optixRayTracer.h"

class BDPT : public OptixRayTracer
{
private:
    LaunchParams<int> launchParams;

public:
    BDPT(const Model* _model, int _w, int _h);

    virtual void render(std::shared_ptr<Camera> camera, std::shared_ptr<Film> film) override;
};