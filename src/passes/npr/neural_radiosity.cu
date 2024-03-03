// #include "neural_radiosity.h"

// REGISTER_RENDER_PASS_CPP(NeuralRadiosity);

// json config = {
//     {
//         "loss", {
//             {"otype", "RelativeL2"},
//         }
//     },
//     {
//         "optimizer", {
//             {"otype", "Adam"},
//             // {"otype", "Shampoo"},
//             {"learning_rate", 1e-2},
//             {"beta1", 0.9f},
//             {"beta2", 0.99f},
//             {"l2_reg", 0.0f},
//         }
//     },
//     {
//         "encoding", {
//             {"otype", "Composite"},
//             {"nested",[
//                 {   // position
//                     {"otype": "HashGrid"},
//                     {"n_dims_to_encode": 3},
//                     {"n_levels": 8},
//                     {"n_features_per_level", 8},
//                     {"base_resolution", 16},
//                     {"per_level_scale", 2.0},
//                     {"interpolation", "Smoothstep"},
//                 },
//                 {   // wo
//                     {"otype": "SphericalHarmonics"},
//                     {"n_dims_to_encode": 3},
//                     {"degree": 3},
//                 },
//                 {   // normal
//                     {"otype": "SphericalHarmonics"},
//                     {"n_dims_to_encode": 3},
//                     {"degree": 3},
//                 }
//             ]}
//         }
//     },
//     {
//         "network", {
//             {"otype", "FullyFusedMLP"},
//             // {"otype", "CutlassMLP"},
//             {"activation", "ReLU"},
//             {"output_activation", "None"},
//             {"n_neurons", 128},
//             {"n_hidden_layers", 4},
//         }
//     },
// }

// void NeuralRadiosity::init()
// {
//     loss = create_loss<precision_t>(config.at("loss"));
//     optimizer = create_optimizer<precision_t>(config.at("optimizer"));
//     network = make_shared<NetworkWithInputEncoding<precision_t>>(
//         n_input_dims, n_output_dims,
//         config.at("encoding"), config.at("network")
//     );
// }

// void NeuralRadiosity::render(shared_ptr<Film> film)
// {
//     PROFILE("NeuralRadiosity");

// }

// void NeuralRadiosity::render_ui()
// {
//     if (ImGui::CollapsingHeader("NeuralRadiosity"))
//     {
//         ImGui::SliderInt("samples per pixel", &samples_per_pixel, 1, 16);
//         ImGui::Combo("type", (int*)&type, "LHS\0RHS\0\0");
//     }
// }