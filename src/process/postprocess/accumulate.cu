#include "accumulate.h"

void AccumulateProcess::render(shared_ptr<Film> film)
{
    float4* pixels = film->get_pixels();
    float4* acc = accumulated.data();
    int cnt = frame_count;
    tcnn::parallel_for_gpu(width * height, [=] __device__(int i) {
        if (cnt == 0)
            acc[i] = pixels[i];
        else
        {
            float w = 1.0f / (float)(cnt + 1);
            acc[i] = acc[i] * (1.0f - w) + pixels[i] * w;
        }
        pixels[i] = acc[i];
    });
    checkCudaErrors(cudaDeviceSynchronize());
    frame_count++;
}

void AccumulateProcess::render_ui()
{
}
