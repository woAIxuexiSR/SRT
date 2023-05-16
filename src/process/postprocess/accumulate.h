#pragma once

#include "process.h"
#include "helper_cuda.h"

class Accumulate : public RenderProcess
{
private:
    REGISTER_RENDER_PROCESS(Accumulate);

    int frame_count{ 0 };
    GPUMemory<float4> accumulated;

public:
    Accumulate() {}

    virtual void resize(int _w, int _h) override;
    virtual void render(shared_ptr<Film> film) override;
    virtual void render_ui() override;

public:
    friend void to_json(json& j, const Accumulate& p) {}
    friend void from_json(const json& j, Accumulate& p) {}
};