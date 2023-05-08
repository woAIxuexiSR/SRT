#pragma once

#include "process.h"
#include "helper_cuda.h"
#include "my_math.h"

enum class ToneMappingType
{
    None,
    Clamp,
    Reinhard,
    Uncharted2,
    ACES
};

class ToneMappingProcess : public RenderProcess
{
private:
    ToneMappingType type;
    float exposure;
    bool use_gamma;

public:
    ToneMappingProcess(ToneMappingType _t, int _w, int _h, shared_ptr<Scene> _s = nullptr)
        : type(_t), exposure(0.0f), use_gamma(true), RenderProcess(_w, _h, _s)
    {}

    virtual void render(shared_ptr<Film> film) override;
    virtual void render_ui() override;
};