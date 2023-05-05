#pragma once

#include "optix_ray_tracer.h"

class MaterialAdjuster: public OptixRayTracer
{
private:
    LaunchParams<Material> params;
    shared_ptr<Material> mat;

public:
    MaterialAdjuster(const Scene* _scene, shared_ptr<Material> _mat);
    virtual void render(shared_ptr<Camera> camera, shared_ptr<Film> film) override;
};


class SimpleShader: public OptixRayTracer
{
private:
    LaunchParams<SimpleShadeType> params;

public:
    SimpleShader(const Scene* _scene, SimpleShadeType _type = SimpleShadeType::Ambient);
    virtual void render(shared_ptr<Camera> camera, shared_ptr<Film> film) override;
};


class PathTracer: public OptixRayTracer
{
private:
    LaunchParams<int> params;

public:
    PathTracer(const Scene* _scene);
    virtual void render(shared_ptr<Camera> camera, shared_ptr<Film> film) override;
};


class LightTracer: public OptixRayTracer
{
private:
    LaunchParams<int> params;

public:
    LightTracer(const Scene* _scene);
    virtual void render(shared_ptr<Camera> camera, shared_ptr<Film> film) override;
};


class DirectLight: public OptixRayTracer
{
private:
    LaunchParams<int> params;

public:
    DirectLight(const Scene* _scene);
    virtual void render(shared_ptr<Camera> camera, shared_ptr<Film> film) override;
};


class BDPT: public OptixRayTracer
{
private:
    LaunchParams<BDPTPath* > params;
    GPUMemory<BDPTPath> light_paths;
    int sqrt_num_light_paths;

public:
    BDPT(const Scene* _scene, int _sqrt_num_light_paths = 1 << 10);
    virtual void render(shared_ptr<Camera> camera, shared_ptr<Film> film) override;
};


// class ReSTIR_DI: public OptixRayTracer
// {
// private:
//     LaunchParams<int> launchParams;

// public:
//     ReSTIR_DI(const Model* _model);

//     void setSamplesPerPixel(int spp) { launchParams.samplesPerPixel = spp; }

//     virtual void render(shared_ptr<Camera> camera, shared_ptr<Film> film) override;
// };


