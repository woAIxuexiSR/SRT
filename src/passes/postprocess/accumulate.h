#pragma once

#include "renderpass.h"
#include "helper_cuda.h"

class Accumulate : public RenderPass
{
private:
    REGISTER_RENDER_PASS(Accumulate);

    int frame_count{ 0 };
    GPUMemory<float4> accumulated;

public:
    Accumulate() {}

    virtual void resize(int _w, int _h) override;
    virtual void render(shared_ptr<Film> film) override;
    virtual void render_ui() override;

public:
    friend void from_json(const json& j, Accumulate& p) {}
};