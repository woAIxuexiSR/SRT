#pragma once

#include "renderpass.h"
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

class ToneMapping : public RenderPass
{
private:
    REGISTER_RENDER_PASS(ToneMapping);

    ToneMappingType type{ ToneMappingType::None };
    float exposure{ 0.0f };
    bool use_gamma{ true };

public:
    ToneMapping() {}

    virtual void render(shared_ptr<Film> film) override;
    virtual void render_ui() override;

public:
    friend void to_json(json& j, const ToneMapping& p)
    {
        j = json{
            { "type", p.type },
            { "exposure", p.exposure },
            { "use_gamma", p.use_gamma }
        };
    }

    friend void from_json(const json& j, ToneMapping& p)
    {
        if (j.is_null()) return;

        p.type = j.value("type", ToneMappingType::None);
        p.exposure = j.value("exposure", 0.0f);
        p.use_gamma = j.value("use_gamma", true);
    }
};