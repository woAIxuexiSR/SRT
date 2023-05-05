#pragma once

#include "process.h"
#include "optix_ray_tracer.h"
#include "my_params.h"

class PathTracer : public RenderProcess
{
private:
    shared_ptr<OptixRayTracer> tracer;

    int max_depth = 16;
    int samples_per_pixel = 4;
    
public:
    PathTracer(int _w, int _h, shared_ptr<Scene> _s);

    virtual void render(shared_ptr<Film> film) override;
    virtual void render_ui() override;
};