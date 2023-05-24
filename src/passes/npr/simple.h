#pragma once

#include "renderpass.h"
#include "optix_ray_tracer.h"
#include "my_params.h"

class Simple : public RenderPass
{
private:
    REGISTER_RENDER_PASS(Simple);

    shared_ptr<OptixRayTracer> tracer;

public:
    Simple() {}

    virtual void set_scene(shared_ptr<Scene> _scene) override;
    virtual void render(shared_ptr<Film> film) override;
    virtual void render_ui() override;

public:
    friend void to_json(json& j, const Simple& p)
    {
        j = json{};
    }

    friend void from_json(const json& j, Simple& p)
    {
        if(j.is_null()) return;
    }

};