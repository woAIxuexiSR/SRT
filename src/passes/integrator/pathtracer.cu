#include "pathtracer.h"

REGISTER_RENDER_PASS_CPP(PathTracer);

void PathTracer::set_scene(shared_ptr<Scene> _scene)
{
    scene = _scene;
    params.resize(1);

    vector<string> ptx_files({ "path_tracer.ptx" });
    tracer = make_shared<OptixRayTracer>(ptx_files, _scene);
}

void PathTracer::render(shared_ptr<Film> film)
{
    PROFILE("PathTracer");
    PathTracerParams host_params;
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
    host_params.use_nee = use_nee;
    host_params.use_mis = use_mis;

    params.copy_from_host(&host_params, 1);
    tracer->trace(width * height, 0, params.data());
    checkCudaErrors(cudaDeviceSynchronize());
}

void PathTracer::render_ui()
{
    if (ImGui::CollapsingHeader("PathTracer"))
    {
        ImGui::SliderInt("samples per pixel", &samples_per_pixel, 1, 8);

        ImGui::PushItemWidth(ImGui::GetWindowWidth() * 0.21f);
        ImGui::SliderInt("max depth", &max_depth, 1, 32);
        ImGui::SameLine();
        ImGui::SliderInt("rr depth", &rr_depth, 1, 16);
        ImGui::PopItemWidth();

        ImGui::Checkbox("use nee", &use_nee);
        ImGui::Checkbox("use mis", &use_mis);
    }
}