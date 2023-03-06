#include "integrator.h"


PathTracer::PathTracer(const Model* _model)
    : OptixRayTracer({ "pathTracer.ptx" }, _model)
    // : OptixRayTracer({ "pathTracer_.ptx" }, _model)
{
    launchParams.traversable = traversable;
    launchParams.light = light;
}

void PathTracer::render(std::shared_ptr<Camera> camera, std::shared_ptr<Film> film)
{
    launchParams.width = film->getWidth();
    launchParams.height = film->getHeight();
    launchParams.colorBuffer = film->getfPtr();
    launchParams.camera = *camera;

    GPUMemory<LaunchParams<int> > launchParamsBuffer;
    launchParamsBuffer.resize_and_copy_from_host(&launchParams, 1);

    OPTIX_CHECK(optixLaunch(
        pipelines[0],
        stream,
        (CUdeviceptr)launchParamsBuffer.data(),
        launchParamsBuffer.size() * sizeof(LaunchParams<int>),
        &sbts[0],
        launchParams.width,
        launchParams.height,
        1));
    checkCudaErrors(cudaDeviceSynchronize());

    launchParams.frameId++;
}


LightTracer::LightTracer(const Model* _model)
    : OptixRayTracer({ "lightTracer.ptx" }, _model)
{
    launchParams.traversable = traversable;
    launchParams.light = light;
}

__global__ void average_radiance(int n_elements, float4* __restrict__ colorBuffer, int* __restrict__ pixelCnt)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n_elements) return;
    if(pixelCnt[i] == 0)
        colorBuffer[i] = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
    // else
    //     colorBuffer[i] /= pixelCnt[i];
}

void LightTracer::render(std::shared_ptr<Camera> camera, std::shared_ptr<Film> film)
{
    film->memset_0();
    launchParams.width = film->getWidth();
    launchParams.height = film->getHeight();
    launchParams.colorBuffer = film->getfPtr();
    launchParams.camera = *camera;

    int n_elements = launchParams.width * launchParams.height;
    pixelCnt.resize(n_elements);
    pixelCnt.memset(0);
    launchParams.extraData = pixelCnt.data();

    GPUMemory<LaunchParams<int*> > launchParamsBuffer;
    launchParamsBuffer.resize_and_copy_from_host(&launchParams, 1);

    OPTIX_CHECK(optixLaunch(
        pipelines[0],
        stream,
        (CUdeviceptr)launchParamsBuffer.data(),
        launchParamsBuffer.size() * sizeof(LaunchParams<int>),
        &sbts[0],
        launchParams.width,
        launchParams.height,
        1));
    checkCudaErrors(cudaDeviceSynchronize());


    tcnn::linear_kernel(average_radiance, 0, 0, n_elements, launchParams.colorBuffer, launchParams.extraData);
    checkCudaErrors(cudaDeviceSynchronize());

    launchParams.frameId++;
}


DirectLight::DirectLight(const Model* _model)
    : OptixRayTracer({ "directLight.ptx" }, _model)
{
    launchParams.traversable = traversable;
    launchParams.light = light;
}

void DirectLight::render(std::shared_ptr<Camera> camera, std::shared_ptr<Film> film)
{
    launchParams.width = film->getWidth();
    launchParams.height = film->getHeight();
    launchParams.colorBuffer = film->getfPtr();
    launchParams.camera = *camera;

    GPUMemory<LaunchParams<int> > launchParamsBuffer;
    launchParamsBuffer.resize_and_copy_from_host(&launchParams, 1);

    OPTIX_CHECK(optixLaunch(
        pipelines[0],
        stream,
        (CUdeviceptr)launchParamsBuffer.data(),
        launchParamsBuffer.size() * sizeof(LaunchParams<int>),
        &sbts[0],
        launchParams.width,
        launchParams.height,
        1));
    checkCudaErrors(cudaDeviceSynchronize());

    launchParams.frameId++;
}


BDPT::BDPT(const Model* _model, int _sqrtNumLightPaths)
    : OptixRayTracer({ "lightPath.ptx", "cameraPath.ptx" }, _model), sqrtNumLightPaths(_sqrtNumLightPaths)
{
    launchParams.traversable = traversable;
    launchParams.light = light;

    lightPaths.resize(sqrtNumLightPaths * sqrtNumLightPaths);
    launchParams.extraData = lightPaths.data();
}

void BDPT::render(std::shared_ptr<Camera> camera, std::shared_ptr<Film> film)
{
    launchParams.width = film->getWidth();
    launchParams.height = film->getHeight();
    launchParams.colorBuffer = film->getfPtr();
    launchParams.camera = *camera;

    GPUMemory<LaunchParams<BDPTPath*> > launchParamsBuffer;
    launchParamsBuffer.resize_and_copy_from_host(&launchParams, 1);

    OPTIX_CHECK(optixLaunch(
        pipelines[0],
        stream,
        (CUdeviceptr)launchParamsBuffer.data(),
        launchParamsBuffer.size() * sizeof(LaunchParams<BDPTPath*>),
        &sbts[0],
        sqrtNumLightPaths,
        sqrtNumLightPaths,
        1));
    checkCudaErrors(cudaDeviceSynchronize());

    OPTIX_CHECK(optixLaunch(
        pipelines[1],
        stream,
        (CUdeviceptr)launchParamsBuffer.data(),
        launchParamsBuffer.size() * sizeof(LaunchParams<BDPTPath*>),
        &sbts[1],
        launchParams.width,
        launchParams.height,
        1));
    checkCudaErrors(cudaDeviceSynchronize());

    launchParams.frameId++;
}

// ReSTIR_DI::ReSTIR_DI(const Model* _model)
//     : OptixRayTracer({ "pathTracer.ptx" }, _model)
// {
//     launchParams.traversable = traversable;
//     launchParams.light = light;
// }

// void ReSTIR_DI::render(std::shared_ptr<Camera> camera, std::shared_ptr<Film> film)
// {
//     launchParams.width = film->getWidth();
//     launchParams.height = film->getHeight();
//     launchParams.colorBuffer = film->getfPtr();
//     launchParams.camera = *camera;

//     GPUMemory<LaunchParams<int> > launchParamsBuffer;
//     launchParamsBuffer.resize_and_copy_from_host(&launchParams, 1);

//     OPTIX_CHECK(optixLaunch(
//         pipelines[0],
//         stream,
//         (CUdeviceptr)launchParamsBuffer.data(),
//         launchParamsBuffer.size() * sizeof(LaunchParams<int>),
//         &sbts[0],
//         launchParams.width,
//         launchParams.height,
//         1));
//     checkCudaErrors(cudaDeviceSynchronize());

//     launchParams.frameId++;
// }