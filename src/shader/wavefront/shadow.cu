#include <optix_device.h>
#include <device_launch_parameters.h>

#include "helper_optix.h"
#include "my_params.h"
#include "my_math.h"

extern "C" __constant__ ShadowParams params;

template<class T>
__device__ inline T* getPRD()
{
    const unsigned u0 = optixGetPayload_0();
    const unsigned u1 = optixGetPayload_1();
    return reinterpret_cast<T*>(unpack_pointer(u0, u1));
}

extern "C" __global__ void __closesthit__radiance() {}

extern "C" __global__ void __closesthit__shadow()
{
    int& prd = *getPRD<int>();
    prd = 0;
}

extern "C" __global__ void __miss__radiance() {}

extern "C" __global__ void __miss__shadow()
{
    int& prd = *getPRD<int>();
    prd = 1;
}

extern "C" __global__ void __raygen__()
{
    const uint3 launch_idx = optixGetLaunchIndex();
    const int idx = launch_idx.x;

    if (idx >= params.shadow_ray_queue->size())
        return;

    const ShadowRayWorkItem& item = params.shadow_ray_queue->fetch(idx);

    int visible;
    uint2 v = pack_pointer(&visible);
    optixTrace(params.traversable, item.ray.pos, item.ray.dir, 1e-3f, item.tmax - 1e-3f, 0.0f,
        OptixVisibilityMask(255), OPTIX_RAY_FLAG_DISABLE_ANYHIT,
        SHADOW_RAY_TYPE, RAY_TYPE_COUNT, SHADOW_RAY_TYPE,
        v.x, v.y);

    if (visible)
        params.pixel_state[item.idx].L += item.beta;
}