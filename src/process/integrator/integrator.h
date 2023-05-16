#pragma once

#include "process.h"
#include "optix_ray_tracer.h"
#include "my_params.h"

class PathTracer : public RenderProcess
{
private:
    REGISTER_RENDER_PROCESS(PathTracer);

    shared_ptr<OptixRayTracer> tracer;
    int max_depth{ 16 };
    int samples_per_pixel{ 4 };

public:
    PathTracer() {}

    virtual void set_scene(shared_ptr<Scene> scene) override;
    virtual void render(shared_ptr<Film> film) override;
    virtual void render_ui() override;

public:
    friend void to_json(json& j, const PathTracer& p)
    {
        j = json{
            {"max_depth", p.max_depth},
            {"samples_per_pixel", p.samples_per_pixel}
        };
    }

    friend void from_json(const json& j, PathTracer& p)
    {
        if (j.is_null()) return;

        p.max_depth = j.value("max_depth", 16);
        p.samples_per_pixel = j.value("samples_per_pixel", 4);
    }
};