#include "integrator.h"

REGISTER_RENDER_PASS_CPP(PathTracer);

void PathTracer::set_scene(shared_ptr<Scene> _scene)
{
    scene = _scene;
    vector<string> ptx_files({ "path_tracer.ptx" });
    tracer = make_shared<OptixRayTracer>(ptx_files, _scene);
}

void PathTracer::render(shared_ptr<Film> film)
{
    PathTracerData data(samples_per_pixel, max_depth, scene->d_scene.light, { 0.0f, 0.0f, 0.0f });
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