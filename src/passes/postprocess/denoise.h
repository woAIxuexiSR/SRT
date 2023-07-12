#pragma once

#include "renderpass.h"
#include "helper_cuda.h"
#include "helper_optix.h"
#include "my_math.h"

class Denoise : public RenderPass
{
private:
    REGISTER_RENDER_PASS(Denoise);

    CUstream stream;
    OptixDeviceContext context;

    OptixDenoiser denoiser;
    OptixDenoiserSizes denoiser_sizes;
    GPUMemory<unsigned char> denoiser_state;
    GPUMemory<unsigned char> denoiser_scratch;

    GPUMemory<float> intensity;
    GPUMemory<float4> denoised;

public:
    Denoise();
    ~Denoise();

    virtual void resize(int _w, int _h) override;
    virtual void render(shared_ptr<Film> film) override;
    virtual void render_ui() override;

public:
    friend void from_json(const json& j, Denoise& p) {}
};