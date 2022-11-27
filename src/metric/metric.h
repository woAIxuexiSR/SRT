#pragma once

#include <cuda_runtime.h>
#include <string>
#include <vector>
#include "helper_math.h"

class Metric
{
private:
    int width, height;
    std::vector<float4> img, ref;
    static constexpr double eps = 1e-6;

public:

    Metric(int _w, int _h, float* _img, float* _ref) : width(_w), height(_h)
    {
        img.resize(width * height);
        ref.resize(width * height);
        for (int i = 0; i < width * height; i++)
        {
            img[i] = make_float4(_img[i * 4], _img[i * 4 + 1], _img[i * 4 + 2], _img[i * 4 + 3]);
            ref[i] = make_float4(_ref[i * 4], _ref[i * 4 + 1], _ref[i * 4 + 2], _ref[i * 4 + 3]);
        }
    }

    Metric(int _w, int _h, const std::vector<float4>& _img, const std::vector<float4>& _ref)
        : width(_w), height(_h), img(_img), ref(_ref)
    { }

    double compute(const std::string& type)
    {
        if(type == "MSE")
            return compute_MSE();
        if(type == "RMSE")
            return compute_RMSE();
        if(type == "relMSE")
            return compute_relMSE();
        if(type == "MAE")
            return compute_MAE();
        if(type == "MAPE")
            return compute_MAPE();
        if(type == "SMAPE")
            return compute_SMAPE();
        if(type == "PSNR")
            return compute_PSNR();
        return 0.0;
    }

    double compute_MSE()
    {
        double mse = 0.0;
        for (int i = 0; i < width * height; i++)
        {
            mse += (img[i].x - ref[i].x) * (img[i].x - ref[i].x);
            mse += (img[i].y - ref[i].y) * (img[i].y - ref[i].y);
            mse += (img[i].z - ref[i].z) * (img[i].z - ref[i].z);
        }
        return mse / (3.0 * width * height);
    }

    double compute_RMSE()
    {
        return std::sqrt(compute_MSE());
    }

    double compute_relMSE()
    {
        double mse = 0.0;
        for (int i = 0; i < width * height; i++)
        {
            mse += (img[i].x - ref[i].x) * (img[i].x - ref[i].x) / (ref[i].x * ref[i].x + eps);
            mse += (img[i].y - ref[i].y) * (img[i].y - ref[i].y) / (ref[i].y * ref[i].y + eps);
            mse += (img[i].z - ref[i].z) * (img[i].z - ref[i].z) / (ref[i].z * ref[i].z + eps);
        }
        return mse / (3.0 * width * height);
    }

    double compute_MAE()
    {
        double mae = 0.0;
        for (int i = 0; i < width * height; i++)
        {
            mae += std::abs(img[i].x - ref[i].x);
            mae += std::abs(img[i].y - ref[i].y);
            mae += std::abs(img[i].z - ref[i].z);
        }
        return mae / (3.0 * width * height);
    }

    double compute_MAPE()
    {
        double mape = 0.0;
        for (int i = 0; i < width * height; i++)
        {
            mape += std::abs(img[i].x - ref[i].x) / (ref[i].x + eps);
            mape += std::abs(img[i].y - ref[i].y) / (ref[i].y + eps);
            mape += std::abs(img[i].z - ref[i].z) / (ref[i].z + eps);
        }
        return mape / (3.0 * width * height);
    }

    double compute_SMAPE()
    {
        double smape = 0.0;
        for (int i = 0; i < width * height; i++)
        {
            smape += std::abs(img[i].x - ref[i].x) / ((ref[i].x + img[i].x) / 2.0 + eps);
            smape += std::abs(img[i].y - ref[i].y) / ((ref[i].y + img[i].y) / 2.0 + eps);
            smape += std::abs(img[i].z - ref[i].z) / ((ref[i].z + img[i].z) / 2.0 + eps);
        }
        return smape / (3.0 * width * height);
    }

    double compute_PSNR()
    {
        return 10.0 * std::log10(1.0 / compute_MSE());
    }

};