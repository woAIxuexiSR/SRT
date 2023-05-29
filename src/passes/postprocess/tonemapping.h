#pragma once

#include "renderpass.h"
#include "helper_cuda.h"
#include "my_math.h"

class ToneMapping : public RenderPass
{
public:
    REGISTER_RENDER_PASS(ToneMapping);
    enum class Type { None, Clamp, Reinhard, Uncharted2, ACES };

private:
    Type type{ Type::None };
    float exposure{ 0.0f };
    bool use_gamma{ true };

public:
    ToneMapping() {}

    virtual void render(shared_ptr<Film> film) override;
    virtual void render_ui() override;

public:
    static string type_to_string(Type type)
    {
        switch (type)
        {
        case Type::None: return "None";
        case Type::Clamp: return "Clamp";
        case Type::Reinhard: return "Reinhard";
        case Type::Uncharted2: return "Uncharted2";
        case Type::ACES: return "ACES";
        default: return "Unknown";
        }
    }

    static Type string_to_type(const string& str)
    {
        if (str == "None") return Type::None;
        if (str == "Clamp") return Type::Clamp;
        if (str == "Reinhard") return Type::Reinhard;
        if (str == "Uncharted2") return Type::Uncharted2;
        if (str == "ACES") return Type::ACES;
        return Type::None;
    }

    friend void to_json(json& j, const ToneMapping& p)
    {
        j = json{
            { "type", ToneMapping::type_to_string(p.type) },
            { "exposure", p.exposure },
            { "use_gamma", p.use_gamma }
        };
    }

    friend void from_json(const json& j, ToneMapping& p)
    {
        if (j.is_null()) return;

        p.type = ToneMapping::string_to_type(j.value("type", "None"));
        p.exposure = j.value("exposure", 0.0f);
        p.use_gamma = j.value("use_gamma", true);
    }
};