#include <optix_device.h>
#include <device_launch_parameters.h>

#include "launch_params/wavefront_params.h"
#include "helper_optix.h"
#include "my_math.h"

extern "C" __constant__ ShadowLaunchParams params;

template<class T>
static __forceinline__ __device__ T* getPRD()
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

extern "C" __global__ void __anyhit__radiance() {}

extern "C" __global__ void __anyhit__shadow() {}

extern "C" __global__ void __miss__radiance() {}

extern "C" __global__ void __miss__shadow()
{
    int& prd = *getPRD<int>();
    prd = 1;
}

extern "C" __global__ void __raygen__()
{
    const uint3 launch_idx = optixGetLaunchIndex();
    int idx = launch_idx.x;
    if(idx >= params.queue->m_size)
        return;

    Ray& ray = params.ray[idx];
    int t = params.dist[idx];
    thrust::pair<unsigned, unsigned> v = pack_pointer(&params.visible[idx]);
    optixTrace(params.traversable, ray.pos, ray.dir, 1e-3f, t - 1e-3f, 0.0f,
        OptixVisibilityMask(255), OPTIX_RAY_FLAG_DISABLE_ANYHIT,
        SHADOW_RAY_TYPE, RAY_TYPE_COUNT, SHADOW_RAY_TYPE,
        v.first, v.second);
}