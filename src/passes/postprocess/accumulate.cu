#include "accumulate.h"

REGISTER_RENDER_PASS_CPP(Accumulate);

void Accumulate::resize(int _w, int _h)
{
    width = _w; height = _h;
    accumulated.resize(_w * _h);
}

void Accumulate::render(shared_ptr<Film> film)
{
    if (!enable || !scene->is_static())
    {
        frame_count = 0;
        return;
    }
    PROFILE("Accumulate");

    float4* pixels = film->get_pixels();
    float4* acc = accumulated.data();
    int cnt = frame_count;
    tcnn::parallel_for_gpu(width * height, [=] __device__(int i) {
        float w = 1.0f / (float)(cnt + 1);
        acc[i] = acc[i] * (1.0f - w) + pixels[i] * w;
        pixels[i] = acc[i];
    });
    checkCudaErrors(cudaDeviceSynchronize());
    frame_count++;
}

void Accumulate::render_ui()
{
    if (ImGui::CollapsingHeader("Accumulate"))
    {
        ImGui::Checkbox("Enable##Accumulate", &enable);
        ImGui::Text("Accumulated Frame Count: %d", frame_count);
    }
}
