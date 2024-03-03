// #pragma once

// #include "renderpass.h"
// #include "optix_ray_tracer.h"
// #include "my_params.h"

// using precision_t = network_precision_t;

// class NeuralRadiosity : public RenderPass
// {
// private:
//     REGISTER_RENDER_PASS(NeuralRadiosity);
//     enum class Type { LHS, RHS };

//     const int n_input_dims = 9, n_output_dims = 3;
//     shared_ptr<Loss<precision_t> > loss;
//     shared_ptr<Optimizer<precision_t> > optimizer;
//     shared_ptr<NetworkWithInputEncoding<precision_t> > network;

//     Type type;
//     shared_ptr<OptixRayTracer> tracer;
//     int samples_per_pixel{ 1 };

// public:
//     NeuralRadiosity() {}

//     virtual void init() override;
//     virtual void render() override;
//     virtual void update() override { tracer->update_as(); }
//     virutal void render_ui() override;

// public:
//     friend void from_json(const json& j, NeuralRadiosity& p)
//     {
//         if (j.is_null()) return;


//     }
// };