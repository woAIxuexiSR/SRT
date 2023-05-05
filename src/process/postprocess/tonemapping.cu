#include "tonemapping.h"

void ToneMappingProcess::render(shared_ptr<Film> film)
{
    if (!enable) return;

    float4* pixels = film->get_pixels();
    ToneMappingType t = type;
    int exp = exposure;
    bool gamma = use_gamma;
    tcnn::parallel_for_gpu(width * height, [=] __device__(int i) {
        float3 color = make_float3(pixels[i]) * pow(2.0f, exp);
        switch (t)
        {
        case ToneMappingType::None:
            break;
        case ToneMappingType::Clamp:
        {
            color = clamp(color, 0.0f, 1.0f);
            break;
        }
        case ToneMappingType::Reinhard:
        {
            float L = luminance(color);
            color = color / (1.0f + L);
            break;
        }
        case ToneMappingType::Uncharted2:
        {
            float A = 0.15f;
            float B = 0.50f;
            float C = 0.10f;
            float D = 0.20f;
            float E = 0.02f;
            float F = 0.30f;
            color = ((color * (A * color + C * B) + D * E) / (color * (A * color + B) + D * F)) - E / F;
            float W = 11.2f;
            float white = ((W * (A * W + C * B) + D * E) / (W * (A * W + B) + D * F)) - E / F;
            color = color / white;
            break;
        }
        case ToneMappingType::ACES:
        {
            color *= 0.6f;
            float a = 2.51f;
            float b = 0.03f;
            float c = 2.43f;
            float d = 0.59f;
            float e = 0.14f;
            color = (color * (a * color + b)) / (color * (c * color + d) + e);
            color = clamp(color, 0.0f, 1.0f);
            break;
        }
        default:
            break;
        }

        if (gamma)
        {
            float v = 1.0f / 2.2f;
            color = make_float3(pow(color.x, v), pow(color.y, v), pow(color.z, v));
        }

        pixels[i] = make_float4(color, 1.0f);
    });
    checkCudaErrors(cudaDeviceSynchronize());
}

void ToneMappingProcess::render_ui()
{

}