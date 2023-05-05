#pragma once

#include "my_math.h"
#include "scene/camera.h"
#include "scene/light.h"
#include "launch_params/launch_params.h"

struct RayWorkItem
{
    float3 beta;
    ReflectionType type;
    int pixel_id;
};

struct MaterialWorkItem
{
    float3 beta;
    Ray wo;
    HitInfo info;
    int pixel_id;
};


class RayWorkQueue
{
public:
    int m_size;
    RayWorkItem* m_item;
    Ray* m_rays;

    __host__ __device__ void set(RayWorkItem* item, Ray* rays)
    {
        m_item = item;
        m_rays = rays;
    }
    __host__ __device__ void reset() { m_size = 0; }
    __host__ __device__ int size() const { return m_size; }
    __device__ int push() { return atomicAdd(&m_size, 1); }
};

class MaterialWorkQueue
{
public:
    int m_size;
    MaterialWorkItem* m_item;

    __host__ __device__ void set(MaterialWorkItem* item)
    {
        m_item = item;
    }
    __host__ __device__ void reset() { m_size = 0; }
    __host__ __device__ int size() const { return m_size; }
    __device__ int push() { return atomicAdd(&m_size, 1); }
};

class ShadowRayWorkQueue
{
public:
    int m_size;
    RayWorkItem* m_item;
    Ray* m_rays;
    int* m_dist;

    __host__ __device__ void set(RayWorkItem* item, Ray* rays, int* dist)
    {
        m_item = item;
        m_rays = rays;
        m_dist = dist;
    }
    __host__ __device__ void reset() { m_size = 0; }
    __host__ __device__ int size() const { return m_size; }
    __device__ int push() { return atomicAdd(&m_size, 1); }
};

struct ClosestLaunchParams
{
    Ray* ray;
    HitInfo* info;
    RayWorkQueue* queue;
    OptixTraversableHandle traversable;
};

struct ShadowLaunchParams
{
    Ray* ray;
    int* dist;
    int* visible;
    ShadowRayWorkQueue* queue;
    OptixTraversableHandle traversable;
};
