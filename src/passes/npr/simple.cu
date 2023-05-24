#include "simple.h"

REGISTER_RENDER_PASS_CPP(Simple);

void Simple::set_scene(shared_ptr<Scene> _scene)
{
    scene = _scene;
    vector<string> ptx_files({ "simple.ptx" });
    tracer = make_shared<OptixRayTracer>(ptx_files, _scene);
}

void Simple::render(shared_ptr<Film> film)
{
    int seed = random_int(0, INT32_MAX);
    Camera camera = *(scene->camera);
    float4* pixels = film->get_pixels();

    AABB aabb = scene->get_aabb();
    float dist = length(aabb.get_pmax() - aabb.get_pmin());
    dist = length(camera.controller.pos - aabb.center()) + dist / 2.0f;

    LaunchParams<float> params(seed, width, height, tracer->get_traversable(),
        camera, pixels, dist);

    GPUMemory<LaunchParams<float>> params_buffer;
    params_buffer.resize_and_copy_from_host(&params, 1);

    tracer->trace(width * height, 0, params_buffer);
    checkCudaErrors(cudaDeviceSynchronize());
}

void Simple::render_ui()
{
}