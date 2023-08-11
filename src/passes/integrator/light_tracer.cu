#include "light_tracer.h"

REGISTER_RENDER_PASS_CPP(LightTracer);

void LightTracer::init()
{
    params.resize(1);

    vector<string> ptx_files({ "light_tracer.ptx" });
    tracer = make_shared<OptixRayTracer>(ptx_files, scene);
}

void LightTracer::render(shared_ptr<Film> film)
{
    PROFILE("LightTracer");

    float4* pixels = film->get_pixels();
    tcnn::parallel_for_gpu(width * height, [=]__device__(int idx) {
        pixels[idx] = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
    });

    static LightTracerParams host_params;
    host_params.seed = random_int(0, INT32_MAX);
    host_params.width = width;
    host_params.height = height;
    host_params.traversable = tracer->get_traversable();
    host_params.camera = *(scene->camera);
    host_params.light = scene->gscene.light_buffer.data();
    host_params.pixels = film->get_pixels();
    host_params.samples_per_pixel = samples_per_pixel;
    host_params.max_depth = max_depth;
    host_params.rr_depth = rr_depth;

    params.copy_from_host(&host_params, 1);
    tracer->trace(width * height, 0, params.data());
    checkCudaErrors(cudaDeviceSynchronize());
}

void LightTracer::render_ui()
{
    if (ImGui::CollapsingHeader("LightTracer"))
    {
        ImGui::SliderInt("samples per pixel", &samples_per_pixel, 1, 8);

        ImGui::PushItemWidth(ImGui::GetWindowWidth() * 0.21f);
        ImGui::SliderInt("max depth", &max_depth, 1, 32);
        ImGui::SameLine();
        ImGui::SliderInt("rr depth", &rr_depth, 1, 16);
        ImGui::PopItemWidth();
    }
}