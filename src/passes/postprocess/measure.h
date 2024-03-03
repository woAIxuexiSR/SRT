#pragma once

#include "renderpass.h"
#include "helper_cuda.h"
#include "texture.h"
#include "metrics.h"

#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <deque>

class Measure : public RenderPass
{
public:
    REGISTER_RENDER_PASS(Measure);
    enum class Option { None, Reference, Error };

    string ref_path;    // absolute path to reference image
    Metrics::Type type{ Metrics::Type::MSE };
    float discard{ 0.1f };      // in percent
    Option option{ Option::None };

    GPUMemory<float4> ref_buffer;
    GPUMemory<float> error_buffer;
    int frame_count{ 0 };

    const int max_history_size{ 50 };
    const int record_interval{ 10 };
    std::deque<float> error_history;

public:
    Measure() {}

    virtual void init() override;
    virtual void render(shared_ptr<Film> film) override;
    virtual void render_ui() override;

public:
    static Metrics::Type string_to_type(const string& str)
    {
        if (str == "MSE") return Metrics::Type::MSE;
        if (str == "MAPE") return Metrics::Type::MAPE;
        if (str == "SMAPE") return Metrics::Type::SMAPE;
        if (str == "RelMSE") return Metrics::Type::RelMSE;
        return Metrics::Type::MSE;
    }
    static string type_to_string(Metrics::Type type)
    {
        switch (type)
        {
        case Metrics::Type::MSE: return "MSE";
        case Metrics::Type::MAPE: return "MAPE";
        case Metrics::Type::SMAPE: return "SMAPE";
        case Metrics::Type::RelMSE: return "RelMSE";
        default: return "MSE";
        }
    }
    static Option string_to_option(const string& str)
    {
        if (str == "None") return Option::None;
        if (str == "Reference") return Option::Reference;
        if (str == "Error") return Option::Error;
        return Option::None;
    }

    friend void from_json(const json& j, Measure& p)
    {
        if (j.find("ref_path") == j.end())
        {
            cout << "ERROR::Measure: no ref_path specified" << endl;
            exit(-1);
        }
        p.ref_path = j.at("ref_path");
        p.type = Measure::string_to_type(j.value("type", "MSE"));
        p.discard = j.value("discard", 0.1f);
        p.option = Measure::string_to_option(j.value("option", "None"));
    }
};