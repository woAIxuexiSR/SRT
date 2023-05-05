#pragma once

#include "process.h"
#include "helper_cuda.h"

class AccumulateProcess : public RenderProcess
{
private:
    int frame_count;
    GPUMemory<float4> accumulated;

public:
    AccumulateProcess(int _w, int _h, shared_ptr<Scene> _s = nullptr)
        : RenderProcess(_w, _h, _s), frame_count(0), accumulated(_w* _h)
    {}

    virtual void render(shared_ptr<Film> film) override;
    virtual void render_ui() override;
};