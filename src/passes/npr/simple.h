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

    virtual void set_scene(shared_ptr<Scene> _scene) override;
    virtual void render(shared_ptr<Film> film) override;
    virtual void render_ui() override;

public:
    static string type_to_string(SimpleParams::Type type)
    {
        switch (type)
        {
        case SimpleParams::Type::Depth: return "Depth";
        case SimpleParams::Type::Normal: return "Normal";
        case SimpleParams::Type::BaseColor: return "BaseColor";
        case SimpleParams::Type::Ambient: return "Ambient";
        case SimpleParams::Type::FaceOrientation: return "FaceOrientation";
        default: return "Unknown";
        }
    }

    static SimpleParams::Type string_to_type(const string& str)
    {
        if (str == "Depth") return SimpleParams::Type::Depth;
        if (str == "Normal") return SimpleParams::Type::Normal;
        if (str == "BaseColor") return SimpleParams::Type::BaseColor;
        if (str == "Ambient") return SimpleParams::Type::Ambient;
        if (str == "FaceOrientation") return SimpleParams::Type::FaceOrientation;
        return SimpleParams::Type::Depth;
    }

    friend void to_json(json& j, const Simple& p)
    {
        j = json{
            {"type", type_to_string(p.type)},
            {"samples_per_pixel", p.samples_per_pixel}
        };
    }

    friend void from_json(const json& j, Simple& p)
    {
        if (j.is_null()) return;

        p.type = Simple::string_to_type(j.value("type", "Depth"));
        p.samples_per_pixel = j.value("samples_per_pixel", 16);
    }

};