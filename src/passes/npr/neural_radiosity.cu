#include "neural_radiosity.h"

REGISTER_RENDER_PASS_CPP(NeuralRadiosity);

// build from string literal
json config = R"({
    "loss": {
        "otype": "RelativeL2"
    },
    "optimizer": {
        "otype": "Adam",
        "learning_rate": 1e-2,
        "beta1": 0.9,
        "beta2": 0.99,
        "l2_reg": 0.0
    },
    "encoding": {
        "otype": "Composite",
        "nested":[
            {
                "otype": "HashGrid",
                "n_dims_to_encode": 3,
                "n_levels": 8,
                "n_features_per_level": 8,
                "base_resolution": 16,
                "per_level_scale": 2.0,
                "interpolation": "Smoothstep"
            },
            {
                "otype": "SphericalHarmonics",
                "n_dims_to_encode": 3,
                "degree": 3
            },
            {
                "otype": "SphericalHarmonics",
                "n_dims_to_encode": 3,
                "degree": 3
            },
            {
                "otype": "Identity",
                "n_dims_to_encode": 3
            }
        ]
    },
    "network": {
        "otype": "CutlassMLP",
        "activation": "ReLU",
        "output_activation": "None",
        "n_neurons": 256,
        "n_hidden_layers": 4
    }
})"_json;

void NeuralRadiosity::load_model()
{
    if (model_path == "") return;
    std::ifstream file(model_path, std::ios::binary);
    if (!file.is_open()) return;
    json params;
    file >> params;
    file.close();
    trainer->deserialize(params);
}

void NeuralRadiosity::save_model()
{
    if (model_path == "") return;
    json params = trainer->serialize();
    std::ofstream file(model_path, std::ios::binary);
    file << params;
    file.close();
}

void NeuralRadiosity::init()
{
    loss.reset(tcnn::create_loss<precision_t>(config.at("loss")));
    optimizer.reset(tcnn::create_optimizer<precision_t>(config.at("optimizer")));
    network = make_shared<tcnn::NetworkWithInputEncoding<precision_t>>(
        n_input_dims, n_output_dims,
        config.at("encoding"), config.at("network")
    );
    trainer = make_shared<tcnn::Trainer<float, precision_t, precision_t> >(network, optimizer, loss);
    load_model();

    uint32_t pixel_num = width * height;
    uint32_t pixel_num_padded = tcnn::next_multiple(pixel_num, tcnn::batch_size_granularity);
    inference_input_buffer.resize(pixel_num_padded * n_input_dims);
    inference_output_buffer.resize(pixel_num_padded * n_output_dims);
    inference_input.set(inference_input_buffer.data(), n_input_dims, pixel_num_padded);
    inference_output.set(inference_output_buffer.data(), n_output_dims, pixel_num_padded);
    mask.resize(pixel_num);
    base.resize(pixel_num);
    weight.resize(pixel_num);
    result.resize(pixel_num);

    lhs_params.resize(1);
    rhs_params.resize(1);
    vector<string> ptx_files({ "nr_lhs.ptx", "nr_rhs.ptx" });
    tracer = make_shared<OptixRayTracer>(ptx_files, scene);
}

void NeuralRadiosity::render_lhs()
{
    PROFILE("RenderLHS");
    static NRLHSParams host_params;
    host_params.seed = random_int(0, INT32_MAX);
    host_params.width = width;
    host_params.height = height;
    host_params.traversable = tracer->get_traversable();
    host_params.camera = *(scene->camera);
    host_params.inference_buffer = inference_input.data();
    host_params.mask = mask.data();
    host_params.base = base.data();

    lhs_params.copy_from_host(&host_params, 1);
    tracer->trace(width * height, 0, lhs_params.data());

    network->inference(inference_input, inference_output);

    float* i_out = inference_output.data();
    bool* mask_ptr = mask.data();
    float3* base_ptr = base.data();
    float3* res = result.data();
    tcnn::parallel_for_gpu(width * height, [=]__device__(int i) {
        float3 color = make_float3(i_out[i * 3], i_out[i * 3 + 1], i_out[i * 3 + 2]);
        res[i] = mask_ptr[i] ? base_ptr[i] : color;
    });
    checkCudaErrors(cudaDeviceSynchronize());
}

void NeuralRadiosity::render_rhs()
{
    PROFILE("RenderRHS");
    static NRRHSParams host_params;
    host_params.seed = random_int(0, INT32_MAX);
    host_params.width = width;
    host_params.height = height;
    host_params.traversable = tracer->get_traversable();
    host_params.camera = *(scene->camera);
    host_params.light = scene->gscene.light_buffer.data();
    host_params.inference_buffer = inference_input.data();
    host_params.mask = mask.data();
    host_params.weight = weight.data();
    host_params.base = base.data();

    rhs_params.copy_from_host(&host_params, 1);
    tracer->trace(width * height, 1, rhs_params.data());

    network->inference(inference_input, inference_output);

    float* i_out = inference_output.data();
    bool* mask_ptr = mask.data();
    float3* base_ptr = base.data();
    float3* weight_ptr = weight.data();
    float3* res = result.data();
    tcnn::parallel_for_gpu(width * height, [=]__device__(int i) {
        float3 color = make_float3(i_out[i * 3], i_out[i * 3 + 1], i_out[i * 3 + 2]);
        res[i] = base_ptr[i];
        if (mask_ptr[i])
            res[i] += weight_ptr[i] * color;
    });
    checkCudaErrors(cudaDeviceSynchronize());
}

void NeuralRadiosity::train()
{
    PROFILE("Train");
    save_model();
}

void NeuralRadiosity::render(shared_ptr<Film> film)
{
    PROFILE("NeuralRadiosity");

    if (train_model)
    {
        assert(online == false);
        train();
    }

    float4* pixels = film->get_pixels();
    tcnn::parallel_for_gpu(width * height, [=]__device__(int i) {
        pixels[i] = make_float4(0.0f);
    });

    float3* res = result.data();
    for (int i = 0; i < samples_per_pixel; i++)
    {
        if (type == Type::LHS)
            render_lhs();
        else
            render_rhs();

        tcnn::parallel_for_gpu(width * height, [=]__device__(int i) {
            float4 color = make_float4(res[i], 1.0f);
            pixels[i] += color;
        });
    }

    int spp = samples_per_pixel;
    tcnn::parallel_for_gpu(width * height, [=]__device__(int i) {
        pixels[i] /= spp;
    });
    checkCudaErrors(cudaDeviceSynchronize());
}

void NeuralRadiosity::render_ui()
{
    if (ImGui::CollapsingHeader("NeuralRadiosity"))
    {
        ImGui::Combo("type", (int*)&type, "LHS\0RHS\0\0");
        ImGui::SliderInt("samples per pixel", &samples_per_pixel, 1, 16);
    }
}