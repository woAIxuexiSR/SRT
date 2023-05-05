#include "integrator.h"

PathTracer::PathTracer(int _w, int _h, shared_ptr<Scene> _s)
    : RenderProcess(_w, _h, _s)
{
    vector<string> ptx_files({ "path_tracer.ptx" });
    tracer = make_shared<OptixRayTracer>(ptx_files, _s);
}

void PathTracer::render(shared_ptr<Film> film)
{
    PathTracerData data(samples_per_pixel, max_depth, tracer->get_light(), { 0.0f, 0.0f, 0.0f });
    int seed = random_int(0, INT32_MAX);
    Camera camera = *(scene->camera);
    float4* pixels = film->get_pixels();
    PathTracerParams params(seed, width, height, tracer->get_traversable(), camera, pixels, data);

    GPUMemory<PathTracerParams> params_buffer;
    params_buffer.resize_and_copy_from_host(&params, 1);

    tracer->trace(width * height, 0, params_buffer);
    checkCudaErrors(cudaDeviceSynchronize());
}

void PathTracer::render_ui()
{
}