#include "pathTracer.h"

PathTracer::PathTracer(const Model* _model, int _w, int _h) 
    : OptixRayTracer({"pathTracer.ptx"}, _model, _w, _h), launchParams(_w, _h, traversable)
{
    
}

void PathTracer::render(std::shared_ptr<Camera> camera, std::shared_ptr<Film> film)
{
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
        width,
        height,
        1));
    checkCudaErrors(cudaDeviceSynchronize());

    launchParams.frameId++;
}