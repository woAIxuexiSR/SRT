#pragma once

#include "definition.h"
#include "my_math.h"
#include <thrust/sort.h>
#include <thrust/reduce.h>

class Metrics
{
public:
    enum class Type { MSE, MAPE, SMAPE, RelMSE };

public:
    Metrics() {}

    static __host__ __device__ float mse(float3 x, float3 ref)
    {
        return dot(x - ref, x - ref) / 3.0f;
    }
    static __host__ __device__ float mape(float3 x, float3 ref)
    {
        float3 t = fabs(x - ref) / (ref + EPSILON);
        return dot(t, { 1.0f, 1.0f, 1.0f }) / 3.0f;
    }
    static __host__ __device__ float smape(float3 x, float3 ref)
    {
        float3 t = 2.0f * fabs(x - ref) / (x + ref + EPSILON);
        return dot(t, { 1.0f, 1.0f, 1.0f }) / 3.0f;
    }
    static __host__ __device__ float relmse(float3 x, float3 ref)
    {
        float3 t = (x - ref) / (ref + EPSILON);
        return dot(t, t) / 3.0f;
    }

    static __host__ __device__ float cal_pixel(float3 x, float3 ref, Type type)
    {
        switch (type)
        {
        case Type::MSE:
            return mse(x, ref);
        case Type::MAPE:
            return mape(x, ref);
        case Type::SMAPE:
            return smape(x, ref);
        case Type::RelMSE:
            return relmse(x, ref);
        default:
            return 0.0f;
        }
    }

    static __host__ float cal_image(float4* x, float4* ref, int num, Type type, float discard = 0.1f)
    {
        GPUMemory<float> error_buffer(num);

        float* error = error_buffer.data();
        tcnn::parallel_for_gpu(num, [=]__device__(int i) {
            float3 x3 = make_float3(x[i]);
            float3 ref3 = make_float3(ref[i]);
            error[i] = cal_pixel(x3, ref3, type);
        });

        thrust::sort(thrust::device, error, error + num);
        int final_num = (int)(num * (1.0f - discard / 100.0f));
        float err = thrust::reduce(thrust::device, error, error + final_num, 0.0f, thrust::plus<float>()) / final_num;
        checkCudaErrors(cudaDeviceSynchronize());

        return err;
    }
};