#pragma once

#include "renderpass.h"
#include "optix_ray_tracer.h"
#include "my_params.h"

class LightTracer : public RenderPass
{
private:
    REGISTER_RENDER_PASS(LightTracer);

    shared_ptr<OptixRayTracer> tracer;
    GPUMemory<LightTracerParams> params;

    int samples_per_pixel{ 1 };
    int max_depth{ 16 };
    int rr_depth{ 4 };

public:
    LightTracer() {}

    virtual void init() override;
    virtual void render(shared_ptr<Film> film) override;
    virtual void update() override { tracer->update_as(); }
    virtual void render_ui() override;

public:
    friend void from_json(const json& j, LightTracer& p)
    {
        if (j.is_null()) return;

        p.samples_per_pixel = j.value("samples_per_pixel", 1);
        p.max_depth = j.value("max_depth", 16);
        p.rr_depth = j.value("rr_depth", 4);
    }
};