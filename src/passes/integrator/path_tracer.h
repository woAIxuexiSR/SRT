#pragma once

#include "renderpass.h"
#include "optix_ray_tracer.h"
#include "my_params.h"

class PathTracer : public RenderPass
{
private:
    REGISTER_RENDER_PASS(PathTracer);

    shared_ptr<OptixRayTracer> tracer;
    GPUMemory<PathTracerParams> params;

    int samples_per_pixel{ 1 };
    int max_depth{ 16 };
    int rr_depth{ 4 };
    bool use_nee{ true };
    bool use_mis{ true };

public:
    PathTracer() {}

    virtual void init() override;
    virtual void render(shared_ptr<Film> film) override;
    virtual void update() override { tracer->update_as(); }
    virtual void render_ui() override;

public:
    friend void from_json(const json& j, PathTracer& p)
    {
        if (j.is_null()) return;

        p.samples_per_pixel = j.value("samples_per_pixel", 1);
        p.max_depth = j.value("max_depth", 16);
        p.rr_depth = j.value("rr_depth", 4);
        p.use_nee = j.value("use_nee", true);
        p.use_mis = j.value("use_mis", true);
    }
};