#include "wavefront.h"

WavefrontRayTracer::WavefrontRayTracer(const Scene* _scene)
    : OptixRayTracer({ "closest.ptx", "shadow.ptx" }, _scene) {}

void WavefrontRayTracer::trace_closest(int n, Ray* ray, HitInfo* info, RayWorkQueue* current)
{
    ClosestLaunchParams params;
    params.ray = ray;
    params.info = info;
    params.traversable = traversable;
    params.queue = current;
    GPUMemory<ClosestLaunchParams> params_buffer;
    params_buffer.resize_and_copy_from_host(&params, 1);

    OPTIX_CHECK(optixLaunch(
        pipelines[0],
        stream,
        (CUdeviceptr)params_buffer.data(),
        params_buffer.size() * sizeof(ClosestLaunchParams),
        &sbts[0],
        n,
        1,
        1));
}

void WavefrontRayTracer::trace_shadow(int n, Ray* ray, int* dist, int* visible, ShadowRayWorkQueue* shadow_queue)
{
    ShadowLaunchParams params;
    params.ray = ray;
    params.dist = dist;
    params.visible = visible;
    params.traversable = traversable;
    params.queue = shadow_queue;
    GPUMemory<ShadowLaunchParams> params_buffer;
    params_buffer.resize_and_copy_from_host(&params, 1);

    OPTIX_CHECK(optixLaunch(
        pipelines[1],
        stream,
        (CUdeviceptr)params_buffer.data(),
        params_buffer.size() * sizeof(ShadowLaunchParams),
        &sbts[1],
        n,
        1,
        1));
}


Wavefront::Wavefront(const Scene* _scene)
{
    ray_tracer = new WavefrontRayTracer(_scene);
}

void Wavefront::init(shared_ptr<Camera> _camera, shared_ptr<Film> _film)
{
    width = _film->get_width();
    height = _film->get_height();
    num_pixels = width * height;
    checkCudaErrors(cudaMalloc(&camera, sizeof(Camera)));
    checkCudaErrors(cudaMemcpy(camera, _camera.get(), sizeof(Camera), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMalloc(&light, sizeof(Light)));
    checkCudaErrors(cudaMemcpy(light, &(ray_tracer->light), sizeof(Light), cudaMemcpyHostToDevice));

    pixels = _film->get_fptr();
    tcnn::parallel_for_gpu(num_pixels, [=, *this] __device__(int idx) {
        pixels[idx] = make_float4(0.0f);
    });

    checkCudaErrors(cudaMalloc(&rng, sizeof(RandomGenerator) * num_pixels));
    tcnn::parallel_for_gpu(num_pixels, [=, *this] __device__(int idx) {
        rng[idx].init(idx / width, idx % width);
    });

    checkCudaErrors(cudaMallocManaged(&ray_queue[0], sizeof(RayWorkQueue)));
    checkCudaErrors(cudaMallocManaged(&ray_queue[1], sizeof(RayWorkQueue)));
    checkCudaErrors(cudaMallocManaged(&shadow_queue, sizeof(ShadowRayWorkQueue)));
    checkCudaErrors(cudaMallocManaged(&material_queue, sizeof(MaterialWorkQueue)));

    checkCudaErrors(cudaMalloc(&(ray_queue[0]->m_item), sizeof(RayWorkItem) * num_pixels));
    checkCudaErrors(cudaMalloc(&(ray_queue[0]->m_rays), sizeof(Ray) * num_pixels));
    checkCudaErrors(cudaMalloc(&(ray_queue[1]->m_item), sizeof(RayWorkItem) * num_pixels));
    checkCudaErrors(cudaMalloc(&(ray_queue[1]->m_rays), sizeof(Ray) * num_pixels));
    checkCudaErrors(cudaMalloc(&(shadow_queue->m_item), sizeof(RayWorkItem) * num_pixels));
    checkCudaErrors(cudaMalloc(&(shadow_queue->m_rays), sizeof(Ray) * num_pixels));
    checkCudaErrors(cudaMalloc(&(shadow_queue->m_dist), sizeof(int) * num_pixels));
    checkCudaErrors(cudaMalloc(&(material_queue->m_item), sizeof(MaterialWorkItem) * num_pixels));

    checkCudaErrors(cudaMalloc(&closest_info, sizeof(HitInfo) * num_pixels));
    checkCudaErrors(cudaMalloc(&shadow_visible, sizeof(int) * num_pixels));

    checkCudaErrors(cudaDeviceSynchronize());
}

void Wavefront::render(shared_ptr<Camera> _camera, shared_ptr<Film> _film)
{
    init(_camera, _film);

    int spp = 128;
    for (int i = 0; i < spp; i++)
    {
        RayWorkQueue* current = ray_queue[0];
        RayWorkQueue* next = ray_queue[1];

        tcnn::parallel_for_gpu(num_pixels, [=, *this] __device__(int idx) {
            RandomGenerator& r = rng[idx];
            float xx = ((idx % width) + r.random_float()) / width;
            float yy = ((idx / width) + r.random_float()) / height;
            current->m_rays[idx] = camera->get_ray(xx, yy);
            RayWorkItem& item = current->m_item[idx];
            item.beta = make_float3(1.0f);
            item.type = ReflectionType::Specular;
            item.pixel_id = idx;

            current->m_size = num_pixels;
        });
        // checkCudaErrors(cudaDeviceSynchronize());
        // tcnn::parallel_for_gpu(1, [=, *this] __device__(int idx) {
        //     current->m_size = num_pixels;
        // });

        for (int depth = 0; depth < 10; depth++)
        {
            ray_tracer->trace_closest(num_pixels, current->m_rays, closest_info, current);

            tcnn::parallel_for_gpu(num_pixels, [=, *this] __device__(int idx) {
                if (idx >= current->m_size) return;

                RayWorkItem& item = current->m_item[idx];
                HitInfo& info = closest_info[idx];
                if (!info.hit)
                    pixels[item.pixel_id] += make_float4(0.0f, 0.0f, 0.0f, 1.0f);
                else if (info.mat->is_emissive())
                {
                    if (info.inner) return;
                    if (item.type == ReflectionType::Specular)
                        pixels[item.pixel_id] += make_float4(item.beta * info.mat->get_emission(), 1.0f);
                }
                else
                {
                    int t = material_queue->push();
                    MaterialWorkItem& mat_item = material_queue->m_item[t];
                    mat_item.beta = item.beta;
                    mat_item.wo = current->m_rays[idx];
                    mat_item.info = info;
                    mat_item.pixel_id = item.pixel_id;
                }
            });

            tcnn::parallel_for_gpu(num_pixels, [=, *this] __device__(int idx) {
                if (idx >= material_queue->m_size) return;

                MaterialWorkItem& item = material_queue->m_item[idx];
                RandomGenerator& r = rng[item.pixel_id];
                HitInfo& info = item.info;

                if (!info.mat->is_specular())
                {
                    LightSample ls = light->sample(r.random_float2());
                    Ray shadow_ray(info.pos, normalize(ls.pos - info.pos));

                    float cos_i = dot(info.normal, shadow_ray.dir);
                    float cos_light = dot(ls.normal, -shadow_ray.dir);
                    if ((cos_light > 0.0f) && (cos_i > 0.0f || info.mat->is_transmissive()))
                    {
                        float t = length(ls.pos - info.pos);
                        float light_pdf = ls.pdf * t * t / cos_light;
                        float3 beta = item.beta * info.mat->eval(shadow_ray.dir, -item.wo.dir, info.normal, info.color, info.inner)
                            * abs(cos_i) * ls.emission / light_pdf;

                        int id = shadow_queue->push();
                        shadow_queue->m_item[id].beta = beta;
                        shadow_queue->m_item[id].pixel_id = item.pixel_id;
                        shadow_queue->m_rays[id] = shadow_ray;
                        shadow_queue->m_dist[id] = t;
                    }
                }

                MaterialSample ms = info.mat->sample(-item.wo.dir, info.normal, r.random_float2(), info.color, info.inner);
                if (ms.pdf <= 1e-5f) return;

                float3 beta = item.beta * ms.f * abs(dot(ms.wi, info.normal)) / ms.pdf;
                if (depth >= 3)
                {
                    float p = max(beta.x, max(beta.y, beta.z));
                    if (r.random_float() > p) return;
                    beta /= p;
                }

                int t = next->push();
                next->m_rays[t] = Ray(info.pos, ms.wi);
                RayWorkItem& next_item = next->m_item[t];
                next_item.beta = beta;
                next_item.type = ms.type;
                next_item.pixel_id = item.pixel_id;
            });

            ray_tracer->trace_shadow(num_pixels, shadow_queue->m_rays, shadow_queue->m_dist, shadow_visible, shadow_queue);

            tcnn::parallel_for_gpu(num_pixels, [=, *this] __device__(int idx) {
                if (idx >= shadow_queue->m_size) return;

                if (!shadow_visible[idx]) return;
                RayWorkItem& item = shadow_queue->m_item[idx];
                pixels[item.pixel_id] += make_float4(item.beta, 1.0f);
            });

            current = next;
            next = ray_queue[depth % 2];

            tcnn::parallel_for_gpu(1, [=, *this] __device__(int idx) {
                next->reset();
                shadow_queue->reset();
                material_queue->reset();
            });

            // cout << depth << " " << current->m_size << endl;
        }

    }

    tcnn::parallel_for_gpu(num_pixels, [=, *this] __device__(int idx) {
        pixels[idx] = make_float4(make_float3(pixels[idx]) / spp, 1.0f);
    });
    checkCudaErrors(cudaDeviceSynchronize());
}