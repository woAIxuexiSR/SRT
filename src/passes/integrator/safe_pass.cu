#include "safe_pass.h"

REGISTER_RENDER_PASS_CPP(SafePass);

void SafePass::init()
{
    pass->set_enable(true);
    pass->resize(width, height);
    pass->set_scene(scene);
    pass->init();

    accumulated.resize(width * height);
}

void SafePass::render(shared_ptr<Film> film)
{
    PROFILE("SafePass");

    int batches = total_samples / batch_size;
    float4* pixels = film->get_pixels();
    float4* acc = accumulated.data();
    
    tcnn::parallel_for_gpu(width * height, [=]__device__(int i) {
        acc[i] = make_float4(0, 0, 0, 0);
    });

    for (int i = 0; i < batches; i++)
    {
        pass->render(film);

        tcnn::parallel_for_gpu(width * height, [=]__device__(int i) {
            acc[i] += pixels[i] / batches;
        });
    }

    tcnn::parallel_for_gpu(width * height, [=]__device__(int i) {
        pixels[i] = acc[i];
    });
    checkCudaErrors(cudaDeviceSynchronize());
}

void SafePass::render_ui()
{
    if (ImGui::CollapsingHeader("SafePass"))
        pass->render_ui();
}