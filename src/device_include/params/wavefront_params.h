#pragma once

#include "helper_optix.h"
#include "scene/camera.h"
#include "scene/light.h"
#include "scene/gmaterial.h"
#include "hit_info.h"

struct PixelState
{
    RandomGenerator rng;
    float3 L;
};

struct RayWorkItem
{
    int idx;

    Ray ray;
    float3 beta;        // throughput
    float pdf;          // scatter pdf
    bool specular;
};

struct MissWorkItem
{
    int idx;

    float3 beta;
    float3 dir;
};

struct HitLightWorkItem
{
    int idx;

    // ray info
    Ray ray;
    float3 beta;
    float pdf;
    bool specular;

    // hit info
    float3 pos;
    float3 normal;
    float2 texcoord;
    const GMaterial* mat;
    int light_id;
};

struct ScatterRayWorkItem
{
    int idx;

    // ray info
    Ray ray;
    float3 beta;

    // hit info
    float3 pos;
    float3 normal;
    float3 color;
    Onb onb;
    const GMaterial* mat;
};

struct ShadowRayWorkItem
{
    int idx;
    Ray ray;
    float tmax;
    float3 beta;
};

template<class T>
class WorkQueue
{
private:
    int m_size;
    T* m_items;

public:
    __host__ __device__ WorkQueue() : m_size(0), m_items(nullptr) {}
    __host__ __device__ WorkQueue(T* items) : m_size(0), m_items(items) {}

    __device__ int size() const { return m_size; }
    __device__ void set_size(int size) { m_size = size; }
    __device__ const T& fetch(int i) const { return m_items[i]; }
    __device__ T& fetch(int i) { return m_items[i]; }
    __device__ void push(const T& item)
    {
        int idx = atomicAdd(&m_size, 1);
        m_items[idx] = item;
    }
};

using RayQueue = WorkQueue<RayWorkItem>;
using MissQueue = WorkQueue<MissWorkItem>;
using HitLightQueue = WorkQueue<HitLightWorkItem>;
using ScatterRayQueue = WorkQueue<ScatterRayWorkItem>;
using ShadowRayQueue = WorkQueue<ShadowRayWorkItem>;

class ClosestParams
{
public:
    OptixTraversableHandle traversable;
    RayQueue* ray_queue;
    MissQueue* miss_queue;
    HitLightQueue* hit_light_queue;
    ScatterRayQueue* scatter_ray_queue;
};

class ShadowParams
{
public:
    OptixTraversableHandle traversable;
    ShadowRayQueue* shadow_ray_queue;
    PixelState* pixel_state;
};