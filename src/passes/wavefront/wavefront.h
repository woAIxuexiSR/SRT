#pragma once

#include "renderpass.h"
#include "optix_ray_tracer.h"
#include "my_params.h"

class Wavefront : public RenderPass
{
private:
    REGISTER_RENDER_PASS(Wavefront);

    shared_ptr<OptixRayTracer> tracer;

    int samples_per_pixel{ 1 };
    int max_depth{ 16 };
    int rr_depth{ 4 };
    bool use_nee{ true };
    bool use_mis{ true };

    // buffers
    int max_queue_size{ 0 };
    GPUMemory<PixelState> pixel_state_buffer;
    GPUMemory<Camera> camera_buffer;
    GPUMemory<ClosestParams> closest_params;
    GPUMemory<ShadowParams> shadow_params;

    GPUMemory<RayWorkItem> ray_work_buffer[2];
    GPUMemory<MissWorkItem> miss_work_buffer;
    GPUMemory<HitLightWorkItem> hit_light_work_buffer;
    GPUMemory<ScatterRayWorkItem> scatter_ray_work_buffer;
    GPUMemory<ShadowRayWorkItem> shadow_ray_work_buffer;

    GPUMemory<RayQueue> ray_queue_buffer[2];
    GPUMemory<MissQueue> miss_queue_buffer;
    GPUMemory<HitLightQueue> hit_light_queue_buffer;
    GPUMemory<ScatterRayQueue> scatter_ray_queue_buffer;
    GPUMemory<ShadowRayQueue> shadow_ray_queue_buffer;

public:
    Wavefront() {}

    /* helper functions */

    void generate_camera_ray();
    void clear_queues(int depth);
    void show_queue_size();
    void handle_miss();
    void handle_hit_light();
    void generate_scatter_ray(int depth);
    void generate_shadow_ray();
    void trace_closest(int depth);
    void trace_shadow();

    /* render pass */

    virtual void init() override;
    virtual void render(shared_ptr<Film> film) override;
    virtual void update() override { tracer->update_as(); }
    virtual void render_ui() override;

public:
    friend void from_json(const json& j, Wavefront& p)
    {
        if (j.is_null()) return;

        p.samples_per_pixel = j.value("samples_per_pixel", 1);
        p.max_depth = j.value("max_depth", 16);
        p.rr_depth = j.value("rr_depth", 4);
        p.use_nee = j.value("use_nee", true);
        p.use_mis = j.value("use_mis", true);
    }
};