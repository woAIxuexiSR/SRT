#pragma once

#include "tiny-cuda-nn/config.h"

#include "renderpass.h"
#include "optix_ray_tracer.h"
#include "my_params.h"


class NeuralRadiosity : public RenderPass
{
private:
    REGISTER_RENDER_PASS(NeuralRadiosity);
    enum class Type { LHS, RHS };

    const int n_input_dims = 12;        // pos, dir, normal, color
    const int n_output_dims = 3;
    shared_ptr<Loss<precision_t> > loss;
    shared_ptr<Optimizer<precision_t> > optimizer;
    shared_ptr<NetworkWithInputEncoding<precision_t> > network;
    shared_ptr<tcnn::Trainer<float, precision_t, precision_t> > trainer;

    GPUMatrix<float> training_input, training_output;
    GPUMemory<float> inference_input_buffer, inference_output_buffer;
    GPUMatrix<float> inference_input, inference_output;
    GPUMemory<bool> mask;
    GPUMemory<float3> base, weight, result;

    shared_ptr<OptixRayTracer> tracer;
    GPUMemory<NRLHSParams> lhs_params;
    GPUMemory<NRRHSParams> rhs_params;

    Type type;
    int samples_per_pixel{ 1 };
    int rhs_samples{ 16 };
    bool train_model{ false };
    string model_path;

public:
    NeuralRadiosity() {}

    void render_lhs();
    void render_rhs();
    void load_model();
    void save_model();
    void train();

    virtual void init() override;
    virtual void render(shared_ptr<Film> film) override;
    virtual void update() override { tracer->update_as(); }
    virtual void render_ui() override;

public:
    static NeuralRadiosity::Type string_to_type(const string& str)
    {
        if (str == "LHS") return NeuralRadiosity::Type::LHS;
        if (str == "RHS") return NeuralRadiosity::Type::RHS;
        return NeuralRadiosity::Type::LHS;
    }

    friend void from_json(const json& j, NeuralRadiosity& p)
    {
        if (j.is_null()) return;

        p.type = NeuralRadiosity::string_to_type(j.value("type", "LHS"));
        p.samples_per_pixel = j.value("samples_per_pixel", 1);
        p.rhs_samples = j.value("rhs_samples", 16);
        p.train_model = j.value("train_model", false);
        p.model_path = j.value("model_path", "");
    }
};