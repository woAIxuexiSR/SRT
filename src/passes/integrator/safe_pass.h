#pragma once

#include "renderpass.h"
#include "optix_ray_tracer.h"
#include "my_params.h"

/*
    SafePass: render with the inner pass, to handle the case that
    extremely large number of samples are used, which may cause the
    floating point overflow.
*/
class SafePass : public RenderPass
{
private:
    REGISTER_RENDER_PASS(SafePass);

    shared_ptr<RenderPass> pass;
    int batch_size{ 4096 };         // inner pass's spp
    int total_samples{ 0 };         // total samples rendered
    GPUMemory<float4> accumulated;

public:
    SafePass() {}

    virtual void init() override;
    virtual void render(shared_ptr<Film> film) override;
    virtual void update() override { pass->update(); }
    virtual void render_ui() override;

public:
    friend void from_json(const json& j, SafePass& p)
    {
        if (j.find("pass") == j.end())
        {
            cout << "ERROR::SafePass: inner pass is missing" << endl;
            exit(-1);
        }

        int batch_size = j.value("batch_size", 4096);
        int spp = 1;

        json pass_config = j.at("pass");
        if (pass_config.find("params") != pass_config.end())
        {
            spp = pass_config["params"].value("samples_per_pixel", 1);
            pass_config["params"]["samples_per_pixel"] = batch_size;
        }
        else
            pass_config["params"] = { {"samples_per_pixel", batch_size} };

        p.batch_size = batch_size;
        p.total_samples = max(spp ,batch_size);
        p.pass = RenderPassFactory::create_pass(pass_config.at("name"), pass_config["params"]);
    }
};