#pragma once

#include "optixRayTracer.h"


class PathTracer: public OptixRayTracer
{
private:
    LaunchParams<int> launchParams;

public:
    PathTracer(const Model* _model);

    void setSamplesPerPixel(int spp) { launchParams.samplesPerPixel = spp; }

    virtual void render(std::shared_ptr<Camera> camera, std::shared_ptr<Film> film) override;
};


class LightTracer: public OptixRayTracer
{
private:
    LaunchParams<int*> launchParams;
    GPUMemory<int> pixelCnt;

public:
    LightTracer(const Model* _model);

    void setSamplesPerPixel(int spp) { launchParams.samplesPerPixel = spp; }

    virtual void render(std::shared_ptr<Camera> camera, std::shared_ptr<Film> film) override;
};


class DirectLight: public OptixRayTracer
{
private:
    LaunchParams<int> launchParams;

public:
    DirectLight(const Model* _model);

    void setSamplesPerPixel(int spp) { launchParams.samplesPerPixel = spp; }

    virtual void render(std::shared_ptr<Camera> camera, std::shared_ptr<Film> film) override;
};


class BDPT: public OptixRayTracer
{
private:
    LaunchParams<BDPTPath* > launchParams;
    GPUMemory<BDPTPath> lightPaths;
    int sqrtNumLightPaths;

public:
    BDPT(const Model* _model, int _sqrtNumLightPaths = 1 << 10);

    void setSamplesPerPixel(int spp) { launchParams.samplesPerPixel = spp; }

    virtual void render(std::shared_ptr<Camera> camera, std::shared_ptr<Film> film) override;
};


// class ReSTIR_DI: public OptixRayTracer
// {
// private:
//     LaunchParams<int> launchParams;

// public:
//     ReSTIR_DI(const Model* _model);

//     void setSamplesPerPixel(int spp) { launchParams.samplesPerPixel = spp; }

//     virtual void render(std::shared_ptr<Camera> camera, std::shared_ptr<Film> film) override;
// };