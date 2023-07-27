#include "denoise.h"

REGISTER_RENDER_PASS_CPP(Denoise);

Denoise::Denoise() : intensity(1)
{
    checkCudaErrors(cudaStreamCreate(&stream));

    CUcontext cuda_context;
    CUresult cu_res = cuCtxGetCurrent(&cuda_context);
    if (cu_res != CUDA_SUCCESS)
    {
        cout << "ERROR::Failed to get current CUDA context" << endl;
        exit(-1);
    }

    OPTIX_CHECK(optixDeviceContextCreate(cuda_context, 0, &context));
}

void Denoise::init()
{
    denoised.resize(width * height);

    OptixDenoiserOptions options = {};
    OPTIX_CHECK(optixDenoiserCreate(context, OPTIX_DENOISER_MODEL_KIND_HDR, &options, &denoiser));

    OPTIX_CHECK(optixDenoiserComputeMemoryResources(denoiser, width, height, &denoiser_sizes));

    denoiser_state.resize(denoiser_sizes.stateSizeInBytes);
    denoiser_scratch.resize(denoiser_sizes.withoutOverlapScratchSizeInBytes);
    OPTIX_CHECK(optixDenoiserSetup(
        denoiser,
        stream,
        width, height,
        (CUdeviceptr)denoiser_state.data(),
        denoiser_sizes.stateSizeInBytes,
        (CUdeviceptr)denoiser_scratch.data(),
        denoiser_sizes.withoutOverlapScratchSizeInBytes
    ));
}

Denoise::~Denoise()
{
    if (denoiser) OPTIX_CHECK(optixDenoiserDestroy(denoiser));
}

void Denoise::render(shared_ptr<Film> film)
{
    if (!enable) return;
    PROFILE("Denoise");

    OptixImage2D input_layer;
    input_layer.data = (CUdeviceptr)film->get_pixels();
    input_layer.width = width;
    input_layer.height = height;
    input_layer.rowStrideInBytes = width * sizeof(float4);
    input_layer.pixelStrideInBytes = sizeof(float4);
    input_layer.format = OPTIX_PIXEL_FORMAT_FLOAT4;

    OptixImage2D output_layer;
    output_layer.data = (CUdeviceptr)denoised.data();
    output_layer.width = width;
    output_layer.height = height;
    output_layer.rowStrideInBytes = width * sizeof(float4);
    output_layer.pixelStrideInBytes = sizeof(float4);
    output_layer.format = OPTIX_PIXEL_FORMAT_FLOAT4;

    OPTIX_CHECK(optixDenoiserComputeIntensity(
        denoiser,
        stream,
        &input_layer,
        (CUdeviceptr)intensity.data(),
        (CUdeviceptr)denoiser_scratch.data(),
        denoiser_sizes.withoutOverlapScratchSizeInBytes
    ));

    OptixDenoiserParams params = {};
    params.denoiseAlpha = 1;
    params.hdrIntensity = (CUdeviceptr)intensity.data();
    params.blendFactor = 0;

    OptixDenoiserGuideLayer guide_layer = {};
    OptixDenoiserLayer layer = {};
    layer.input = input_layer;
    layer.output = output_layer;

    OPTIX_CHECK(optixDenoiserInvoke(
        denoiser,
        stream,
        &params,
        (CUdeviceptr)denoiser_state.data(),
        denoiser_sizes.stateSizeInBytes,
        &guide_layer,
        &layer,
        1, 0, 0,
        (CUdeviceptr)denoiser_scratch.data(),
        denoiser_sizes.withoutOverlapScratchSizeInBytes
    ));

    checkCudaErrors(cudaMemcpy(film->get_pixels(), denoised.data(), width * height * sizeof(float4), cudaMemcpyDeviceToDevice));
}

void Denoise::render_ui()
{
    if (ImGui::CollapsingHeader("Denoise"))
        ImGui::Checkbox("Enable##Denoise", &enable);
}