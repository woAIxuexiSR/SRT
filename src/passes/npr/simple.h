#pragma once

#include "renderpass.h"
#include "optix_ray_tracer.h"
#include "my_params.h"

class Simple : public RenderPass
{
private:
    REGISTER_RENDER_PASS(Simple);

    shared_ptr<OptixRayTracer> tracer;
    GPUMemory<SimpleParams> params;

    SimpleParams::Type type;
    int samples_per_pixel{ 16 };

public:
    Simple() {}

    virtual void init() override;
    virtual void render(shared_ptr<Film> film) override;
    virtual void update() override { tracer->update_as(); }
    virtual void render_ui() override;

public:
    static SimpleParams::Type string_to_type(const string& str)
    {
        if (str == "Depth") return SimpleParams::Type::Depth;
        if (str == "Normal") return SimpleParams::Type::Normal;
        if (str == "BaseColor") return SimpleParams::Type::BaseColor;
        if (str == "Ambient") return SimpleParams::Type::Ambient;
        if (str == "FaceOrientation") return SimpleParams::Type::FaceOrientation;
        return SimpleParams::Type::Depth;
    }

    friend void from_json(const json& j, Simple& p)
    {
        if (j.is_null()) return;

        p.type = Simple::string_to_type(j.value("type", "Depth"));
        p.samples_per_pixel = j.value("samples_per_pixel", 16);
    }

};