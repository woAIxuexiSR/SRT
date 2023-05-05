#pragma once

#include "process.h"
#include "helper_cuda.h"
#include "helper_optix.h"
#include "my_math.h"

class DenoiseProcess : public RenderProcess
{
private:
    CUstream stream;
    OptixDeviceContext context;
    
    OptixDenoiser denoiser;
    OptixDenoiserSizes denoiser_sizes;
    GPUMemory<unsigned char> denoiser_state;
    GPUMemory<unsigned char> denoiser_scratch;

    GPUMemory<float> intensity;
    GPUMemory<float4> denoised;

public:
    DenoiseProcess(int _w, int _h, shared_ptr<Scene> _s = nullptr);
    ~DenoiseProcess();

    virtual void render(shared_ptr<Film> film) override;
    virtual void render_ui() override;
};