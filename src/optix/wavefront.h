#pragma once

#include "optix_ray_tracer.h"
#include "launch_params/wavefront_params.h"


class WavefrontRayTracer : public OptixRayTracer
{
public:
    WavefrontRayTracer(const Scene* _scene);
    virtual void render(shared_ptr<Camera> _camera, shared_ptr<Film> _film) override {}

    void trace_closest(int n, Ray* ray, HitInfo* info, RayWorkQueue* current);
    void trace_shadow(int n, Ray* ray, int* dist, int* visible, ShadowRayWorkQueue* shadow_queue);
};


class Wavefront
{
private:
    WavefrontRayTracer* ray_tracer;

    int width, height;
    int num_pixels;
    Camera* camera;
    Light* light;
    float4* pixels;

    RandomGenerator* rng;

    RayWorkQueue* ray_queue[2];
    HitInfo* closest_info;
    ShadowRayWorkQueue* shadow_queue;
    int* shadow_visible;
    MaterialWorkQueue* material_queue;



public:
    Wavefront(const Scene* _scene);
    void init(shared_ptr<Camera> _camera, shared_ptr<Film> _film);
    void render(shared_ptr<Camera> _camera, shared_ptr<Film> _film);
};