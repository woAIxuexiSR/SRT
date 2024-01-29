#include "wavefront.h"

REGISTER_RENDER_PASS_CPP(Wavefront);

void Wavefront::init()
{
    vector<string> ptx_files({ "closest.ptx", "shadow.ptx" });
    tracer = make_shared<OptixRayTracer>(ptx_files, scene);

    max_queue_size = width * height;
    pixel_state_buffer.resize(max_queue_size);
    camera_buffer.resize(1);
    closest_params.resize(1);
    shadow_params.resize(1);

    for (int i = 0; i < 2; i++)
    {
        ray_work_buffer[i].resize(max_queue_size);
        RayQueue ray_queue(ray_work_buffer[i].data());
        ray_queue_buffer[i].resize_and_copy_from_host(&ray_queue, 1);
    }

    miss_work_buffer.resize(max_queue_size);
    MissQueue miss_queue(miss_work_buffer.data());
    miss_queue_buffer.resize_and_copy_from_host(&miss_queue, 1);

    hit_light_work_buffer.resize(max_queue_size);
    HitLightQueue hit_light_queue(hit_light_work_buffer.data());
    hit_light_queue_buffer.resize_and_copy_from_host(&hit_light_queue, 1);

    scatter_ray_work_buffer.resize(max_queue_size);
    ScatterRayQueue scatter_ray_queue(scatter_ray_work_buffer.data());
    scatter_ray_queue_buffer.resize_and_copy_from_host(&scatter_ray_queue, 1);

    shadow_ray_work_buffer.resize(max_queue_size);
    ShadowRayQueue shadow_ray_queue(shadow_ray_work_buffer.data());
    shadow_ray_queue_buffer.resize_and_copy_from_host(&shadow_ray_queue, 1);
}

void Wavefront::generate_camera_ray()
{
    PROFILE("Generate Camera Ray");
    PixelState* pixel_state = pixel_state_buffer.data();
    RayQueue* ray_queue = ray_queue_buffer[0].data();
    Camera* camera = camera_buffer.data();
    int w = width, h = height;
    tcnn::parallel_for_gpu(max_queue_size, [=]__device__(int i) {
        RandomGenerator& rng = pixel_state[i].rng;
        float xx = ((i % w) + rng.random_float()) / w;
        float yy = ((i / w) + rng.random_float()) / h;

        RayWorkItem& item = ray_queue->fetch(i);
        item.idx = i;
        item.ray = camera->get_ray(xx, yy, rng);
        item.beta = make_float3(1.0f);
        item.pdf = 1.0f;
        item.specular = true;
    });

    int queue_size = max_queue_size;
    tcnn::parallel_for_gpu(1, [=]__device__(int i) {
        ray_queue->set_size(queue_size);
    });
}

void Wavefront::clear_queues(int depth)
{
    PROFILE("Clear Queues");
    RayQueue* ray_queue = ray_queue_buffer[(depth + 1) & 1].data();
    MissQueue* miss_queue = miss_queue_buffer.data();
    HitLightQueue* hit_light_queue = hit_light_queue_buffer.data();
    ScatterRayQueue* scatter_ray_queue = scatter_ray_queue_buffer.data();
    ShadowRayQueue* shadow_ray_queue = shadow_ray_queue_buffer.data();
    tcnn::parallel_for_gpu(1, [=]__device__(int i) {
        ray_queue->set_size(0);
        miss_queue->set_size(0);
        hit_light_queue->set_size(0);
        scatter_ray_queue->set_size(0);
        shadow_ray_queue->set_size(0);
    });
}

void Wavefront::show_queue_size()
{
    RayQueue* ray_queue0 = ray_queue_buffer[0].data();
    RayQueue* ray_queue1 = ray_queue_buffer[1].data();
    MissQueue* miss_queue = miss_queue_buffer.data();
    HitLightQueue* hit_light_queue = hit_light_queue_buffer.data();
    ScatterRayQueue* scatter_ray_queue = scatter_ray_queue_buffer.data();
    ShadowRayQueue* shadow_ray_queue = shadow_ray_queue_buffer.data();
    tcnn::parallel_for_gpu(1, [=]__device__(int i) {
        printf("%d ", ray_queue0->size());
        printf("%d ", ray_queue1->size());
        printf("%d ", miss_queue->size());
        printf("%d ", hit_light_queue->size());
        printf("%d ", scatter_ray_queue->size());
        printf("%d\n", shadow_ray_queue->size());
    });
    checkCudaErrors(cudaDeviceSynchronize());
}

void Wavefront::handle_miss()
{
    PROFILE("Handle Miss");
    PixelState* pixel_state = pixel_state_buffer.data();
    MissQueue* miss_queue = miss_queue_buffer.data();
    Light* light = scene->gscene.light_buffer.data();
    tcnn::parallel_for_gpu(max_queue_size, [=]__device__(int i) {
        if (i >= miss_queue->size()) return;
        const MissWorkItem& item = miss_queue->fetch(i);
        pixel_state[item.idx].L += item.beta * light->environment_emission(item.dir);
    });
}

void Wavefront::handle_hit_light()
{
    PROFILE("Handle Hit Light");
    PixelState* pixel_state = pixel_state_buffer.data();
    HitLightQueue* hit_light_queue = hit_light_queue_buffer.data();
    Light* light = scene->gscene.light_buffer.data();
    bool nee = use_nee, mis = use_mis;
    tcnn::parallel_for_gpu(max_queue_size, [=]__device__(int i) {
        if (i >= hit_light_queue->size()) return;
        const HitLightWorkItem& item = hit_light_queue->fetch(i);

        float cos_i = -dot(item.normal, item.ray.dir);
        if (cos_i < 0.0f) return;

        float mis_weight = 1.0f;
        if (nee && !item.specular)
        {
            mis_weight = 0.0f;
            if (mis)
            {
                float t2 = dot(item.pos - item.ray.pos, item.pos - item.ray.pos);
                float light_pdf = light->sample_pdf(item.light_id) * t2 / cos_i;
                mis_weight = item.pdf / (item.pdf + light_pdf);
            }
        }
        pixel_state[item.idx].L += item.beta * item.mat->emission(item.texcoord) * mis_weight;
    });
}

void Wavefront::generate_scatter_ray(int depth)
{
    PROFILE("Generate Scatter Ray");
    PixelState* pixel_state = pixel_state_buffer.data();
    ScatterRayQueue* scatter_ray_queue = scatter_ray_queue_buffer.data();
    RayQueue* ray_queue = ray_queue_buffer[(depth + 1) & 1].data();
    int rr = rr_depth;
    tcnn::parallel_for_gpu(max_queue_size, [=]__device__(int i) {
        if (i >= scatter_ray_queue->size()) return;
        const ScatterRayWorkItem& item = scatter_ray_queue->fetch(i);
        RandomGenerator& rng = pixel_state[item.idx].rng;
        Ray ray = item.ray;

        BxDFSample ms = item.mat->sample(-ray.dir, rng.random_float2(), item.onb, item.color);
        if (ms.pdf <= 1e-5f) return;
        float3 beta = item.beta * ms.f / ms.pdf;
        if (depth >= rr)
        {
            float p = max(beta.x, max(beta.y, beta.z));
            if (rng.random_float() > p) return;
            beta /= p;
        }

        RayWorkItem ray_item;
        ray_item.idx = item.idx;
        ray_item.ray = Ray(item.pos, ms.wi);
        ray_item.beta = beta;
        ray_item.pdf = ms.pdf;
        ray_item.specular = item.mat->is_specular();
        ray_queue->push(ray_item);
    });
}

void Wavefront::generate_shadow_ray()
{
    PROFILE("Generate Shadow Ray");
    PixelState* pixel_state = pixel_state_buffer.data();
    ScatterRayQueue* scatter_ray_queue = scatter_ray_queue_buffer.data();
    ShadowRayQueue* shadow_ray_queue = shadow_ray_queue_buffer.data();
    Light* light = scene->gscene.light_buffer.data();
    bool mis = use_mis;
    tcnn::parallel_for_gpu(max_queue_size, [=]__device__(int i) {
        if (i >= scatter_ray_queue->size()) return;
        const ScatterRayWorkItem& item = scatter_ray_queue->fetch(i);
        RandomGenerator& rng = pixel_state[item.idx].rng;
        Ray ray = item.ray;

        if (!item.mat->is_specular())
        {
            LightSample ls = light->sample(rng.random_float2());
            Ray shadow_ray(item.pos, normalize(ls.pos - item.pos));

            float cos_i = dot(item.normal, -ray.dir);
            float3 normal = (cos_i < 0.0f) ? 0.0f - item.normal : item.normal;
            float cos_o = dot(normal, shadow_ray.dir);
            float cos_light = dot(ls.normal, -shadow_ray.dir);
            if ((cos_light > 0.0f) && (cos_o > 0.0f || item.mat->is_transmissive()))
            {
                float t = length(ls.pos - item.pos);
                float light_pdf = ls.pdf * t * t / cos_light;
                float mis_weight = 1.0f;
                if (mis)
                {
                    float mat_pdf = item.mat->sample_pdf(shadow_ray.dir, -ray.dir, item.onb, item.color);
                    mis_weight = light_pdf / (light_pdf + mat_pdf);
                }
                float3 beta = item.beta * item.mat->eval(shadow_ray.dir, -ray.dir, item.onb, item.color)
                    * abs(cos_o) * ls.emission * mis_weight / light_pdf;

                ShadowRayWorkItem shadow_item;
                shadow_item.idx = item.idx;
                shadow_item.ray = shadow_ray;
                shadow_item.tmax = t;
                shadow_item.beta = beta;
                shadow_ray_queue->push(shadow_item);
            }
        }
    });
}

void Wavefront::trace_closest(int depth)
{
    PROFILE("Trace Closest");
    static ClosestParams host_params;
    host_params.traversable = tracer->get_traversable();
    host_params.ray_queue = ray_queue_buffer[depth & 1].data();
    host_params.miss_queue = miss_queue_buffer.data();
    host_params.hit_light_queue = hit_light_queue_buffer.data();
    host_params.scatter_ray_queue = scatter_ray_queue_buffer.data();

    closest_params.copy_from_host(&host_params, 1);
    tracer->trace(max_queue_size, 0, closest_params.data());
}

void Wavefront::trace_shadow()
{
    PROFILE("Trace Shadow");
    static ShadowParams host_params;
    host_params.traversable = tracer->get_traversable();
    host_params.shadow_ray_queue = shadow_ray_queue_buffer.data();
    host_params.pixel_state = pixel_state_buffer.data();

    shadow_params.copy_from_host(&host_params, 1);
    tracer->trace(max_queue_size, 1, shadow_params.data());
}

void Wavefront::render(shared_ptr<Film> film)
{
    PROFILE("Wavefront");

    {
        PROFILE("Init");
        camera_buffer.copy_from_host(scene->camera.get(), 1);

        PixelState* pixel_state = pixel_state_buffer.data();
        int seed = random_int(0, INT32_MAX);
        tcnn::parallel_for_gpu(max_queue_size, [=]__device__(int i) {
            pixel_state[i].rng.init(seed + i, 0);
            pixel_state[i].L = make_float3(0.0f);
        });
    }

    for (int i = 0; i < samples_per_pixel; i++)
    {
        generate_camera_ray();
        // show_queue_size();
        for (int depth = 0; ; depth++)
        {
            clear_queues(depth);
            trace_closest(depth);


            handle_miss();
            handle_hit_light();

            if (depth == max_depth) break;

            generate_scatter_ray(depth);
            if (use_nee)
            {
                generate_shadow_ray();
                trace_shadow();
            }
        }
    }

    {
        PROFILE("Copy Result");
        PixelState* pixel_state = pixel_state_buffer.data();
        float4* pixels = film->get_pixels();
        int spp = samples_per_pixel;
        tcnn::parallel_for_gpu(max_queue_size, [=]__device__(int i) {
            pixels[i] = make_float4(pixel_state[i].L / spp, 1.0f);
        });
        checkCudaErrors(cudaDeviceSynchronize());
    }
}

void Wavefront::render_ui()
{
    if (ImGui::CollapsingHeader("Wavefront"))
    {
        ImGui::SliderInt("samples per pixel", &samples_per_pixel, 1, 8);

        ImGui::PushItemWidth(ImGui::GetWindowWidth() * 0.21f);
        ImGui::SliderInt("max depth", &max_depth, 1, 32);
        ImGui::SameLine();
        ImGui::SliderInt("rr depth", &rr_depth, 1, 16);
        ImGui::PopItemWidth();

        ImGui::Checkbox("use nee", &use_nee);
        ImGui::Checkbox("use mis", &use_mis);
    }
}