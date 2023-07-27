#include "simple.h"

REGISTER_RENDER_PASS_CPP(Simple);

void Simple::init()
{
    params.resize(1);

    vector<string> ptx_files({ "simple.ptx" });
    tracer = make_shared<OptixRayTracer>(ptx_files, scene);
}

void Simple::render(shared_ptr<Film> film)
{
    PROFILE("Simple");
    SimpleParams host_params;
    host_params.seed = random_int(0, INT32_MAX);
    host_params.width = width;
    host_params.height = height;
    host_params.traversable = tracer->get_traversable();
    host_params.camera = *(scene->camera);
    host_params.pixels = film->get_pixels();
    host_params.type = type;
    host_params.samples_per_pixel = samples_per_pixel;

    float3 pos = scene->camera->controller.pos;
    host_params.min_depth = scene->aabb.min_distance(pos);
    host_params.max_depth = scene->aabb.max_distance(pos);

    params.copy_from_host(&host_params, 1);
    tracer->trace(width * height, 0, params.data());
    checkCudaErrors(cudaDeviceSynchronize());
}

void Simple::render_ui()
{
    if (ImGui::CollapsingHeader("Simple"))
    {
        ImGui::SliderInt("samples per pixel", &samples_per_pixel, 1, 16);
        ImGui::Combo("type", (int*)&type, "Depth\0Normal\0BaseColor\0Ambient\0FaceOrientation\0\0");
    }
}