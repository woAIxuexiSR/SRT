#include "integrator.h"

MaterialAdjuster::MaterialAdjuster(const Scene* _scene, shared_ptr<Material> _mat)
    : OptixRayTracer({ "material_adjuster.ptx" }, _scene), mat(_mat)
{
    params.traversable = traversable;
    params.light = light;
}

void MaterialAdjuster::render(shared_ptr<Camera> camera, shared_ptr<Film> film)
{
    params.spp = spp;
    params.background = background;
    params.width = film->get_width();
    params.height = film->get_height();
    params.buffer = film->get_fptr();
    params.camera = *camera;
    params.extra = *mat;

    GPUMemory<LaunchParams<Material> > params_buffer;
    params_buffer.resize_and_copy_from_host(&params, 1);

    OPTIX_CHECK(optixLaunch(
        pipelines[0],
        stream,
        (CUdeviceptr)params_buffer.data(),
        params_buffer.size() * sizeof(LaunchParams<Material>),
        &sbts[0],
        params.width,
        params.height,
        1));
    checkCudaErrors(cudaDeviceSynchronize());

    params.frame++;
}


SimpleShader::SimpleShader(const Scene* _scene, SimpleShadeType _type)
    : OptixRayTracer({ "simple_shader.ptx" }, _scene)
{
    params.traversable = traversable;
    params.light = light;
    params.extra = _type;
}

void SimpleShader::render(shared_ptr<Camera> camera, shared_ptr<Film> film)
{
    params.spp = spp;
    params.background = background;
    params.width = film->get_width();
    params.height = film->get_height();
    params.buffer = film->get_fptr();
    params.camera = *camera;

    GPUMemory<LaunchParams<SimpleShadeType> > params_buffer;
    params_buffer.resize_and_copy_from_host(&params, 1);

    OPTIX_CHECK(optixLaunch(
        pipelines[0],
        stream,
        (CUdeviceptr)params_buffer.data(),
        params_buffer.size() * sizeof(LaunchParams<SimpleShadeType>),
        &sbts[0],
        params.width,
        params.height,
        1));
    checkCudaErrors(cudaDeviceSynchronize());

    params.frame++;
}

PathTracer::PathTracer(const Scene* _scene)
    : OptixRayTracer({ "path_tracer.ptx" }, _scene)
    // : OptixRayTracer({ "__path_tracer.ptx" }, _scene)
{
    params.traversable = traversable;
    params.light = light;
}

void PathTracer::render(shared_ptr<Camera> camera, shared_ptr<Film> film)
{
    params.spp = spp;
    params.background = background;
    params.width = film->get_width();
    params.height = film->get_height();
    params.buffer = film->get_fptr();
    params.camera = *camera;

    GPUMemory<LaunchParams<int> > params_buffer;
    params_buffer.resize_and_copy_from_host(&params, 1);

    OPTIX_CHECK(optixLaunch(
        pipelines[0],
        stream,
        (CUdeviceptr)params_buffer.data(),
        params_buffer.size() * sizeof(LaunchParams<int>),
        &sbts[0],
        params.width,
        params.height,
        1));
    checkCudaErrors(cudaDeviceSynchronize());

    params.frame++;
}

LightTracer::LightTracer(const Scene* _scene)
    : OptixRayTracer({ "light_tracer.ptx" }, _scene)
{
    params.traversable = traversable;
    params.light = light;
}

void LightTracer::render(shared_ptr<Camera> camera, shared_ptr<Film> film)
{
    film->memset_f0();
    params.spp = spp;
    params.background = background;
    params.width = film->get_width();
    params.height = film->get_height();
    params.buffer = film->get_fptr();
    params.camera = *camera;

    GPUMemory<LaunchParams<int> > params_buffer;
    params_buffer.resize_and_copy_from_host(&params, 1);

    OPTIX_CHECK(optixLaunch(
        pipelines[0],
        stream,
        (CUdeviceptr)params_buffer.data(),
        params_buffer.size() * sizeof(LaunchParams<int>),
        &sbts[0],
        params.width,
        params.height,
        1));
    checkCudaErrors(cudaDeviceSynchronize());

    params.frame++;
}

DirectLight::DirectLight(const Scene* _scene)
    : OptixRayTracer({ "direct_light.ptx" }, _scene)
{
    params.traversable = traversable;
    params.light = light;
}

void DirectLight::render(shared_ptr<Camera> camera, shared_ptr<Film> film)
{
    params.spp = spp;
    params.background = background;
    params.width = film->get_width();
    params.height = film->get_height();
    params.buffer = film->get_fptr();
    params.camera = *camera;

    GPUMemory<LaunchParams<int> > params_buffer;
    params_buffer.resize_and_copy_from_host(&params, 1);

    OPTIX_CHECK(optixLaunch(
        pipelines[0],
        stream,
        (CUdeviceptr)params_buffer.data(),
        params_buffer.size() * sizeof(LaunchParams<int>),
        &sbts[0],
        params.width,
        params.height,
        1));
    checkCudaErrors(cudaDeviceSynchronize());

    params.frame++;
}

BDPT::BDPT(const Scene* _scene, int _sqrt_num_light_paths)
    : OptixRayTracer({ "lightPath.ptx", "cameraPath.ptx" }, _scene), sqrt_num_light_paths(_sqrt_num_light_paths)
{
    params.traversable = traversable;
    params.light = light;

    light_paths.resize(sqrt_num_light_paths * sqrt_num_light_paths);
    params.extra = light_paths.data();
}

void BDPT::render(shared_ptr<Camera> camera, shared_ptr<Film> film)
{
    params.spp = spp;
    params.background = background;
    params.width = film->get_width();
    params.height = film->get_height();
    params.buffer = film->get_fptr();
    params.camera = *camera;

    GPUMemory<LaunchParams<BDPTPath*> > params_buffer;
    params_buffer.resize_and_copy_from_host(&params, 1);

    OPTIX_CHECK(optixLaunch(
        pipelines[0],
        stream,
        (CUdeviceptr)params_buffer.data(),
        params_buffer.size() * sizeof(LaunchParams<BDPTPath*>),
        &sbts[0],
        sqrt_num_light_paths,
        sqrt_num_light_paths,
        1));
    checkCudaErrors(cudaDeviceSynchronize());

    OPTIX_CHECK(optixLaunch(
        pipelines[1],
        stream,
        (CUdeviceptr)params_buffer.data(),
        params_buffer.size() * sizeof(LaunchParams<BDPTPath*>),
        &sbts[1],
        params.width,
        params.height,
        1));
    checkCudaErrors(cudaDeviceSynchronize());

    params.frame++;
}

// ReSTIR_DI::ReSTIR_DI(const Model* _scene)
//     : OptixRayTracer({ "pathTracer.ptx" }, _scene)
// {
//     launchParams.traversable = traversable;
//     launchParams.light = light;
// }

// void ReSTIR_DI::render(shared_ptr<Camera> camera, shared_ptr<Film> film)
// {
//     launchParams.width = film->get_width();
//     launchParams.height = film->get_height();
//     launchParams.buffer = film->get_fptr();
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

//     launchParams.frame++;
// }