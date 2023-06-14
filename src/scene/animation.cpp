#include "animation.h"

float3 Animation::interpolate_float3(float3 a, float3 b, float factor) const
{
    if(itype == Interpolation::Linear)
        return a * (1.0f - factor) + b * factor;
    else if(itype == Interpolation::Cubic)
    {
        factor = cbrt((factor - 0.5f) * 0.25f) + 0.5f;
        return a * (1.0f - factor) + b * factor;
    }
    return { 0.0f, 0.0f, 0.0f };
}

float Animation::extrapolate_time(float t) const
{
    if(etype == Extrapolation::Clamp)
        return clamp(t, 0.0f, duration);
    else if(etype == Extrapolation::Repeat)
        return fmod(t, duration);
    else if(etype == Extrapolation::Mirror)
    {
        float t2 = fmod(t, duration * 2.0f);
        if(t2 > duration)
            t2 = duration * 2.0f - t2;
        return t2;
    }
    return 0.0f;
}

float3 Animation::interpolate_translation(float t) const
{
    if(translations.size() == 0) return { 0.0f, 0.0f, 0.0f };
    if(translations.size() == 1) return translations[0].translation;

    int idx = 0;
    for(; idx < translations.size() - 1; idx++)
    {
        if(t <= translations[idx + 1].time_stamp)
            break;
    }
    float factor = (t - translations[idx].time_stamp) / (translations[idx + 1].time_stamp - translations[idx].time_stamp);
    return interpolate_float3(translations[idx].translation, translations[idx + 1].translation, factor);
}

Quaternion Animation::interpolate_rotation(float t) const
{
    if(rotations.size() == 0) return Quaternion::Identity();
    if(rotations.size() == 1) return rotations[0].rotation;

    int idx = 0;
    for(; idx < rotations.size() - 1; idx++)
    {
        if(t <= rotations[idx + 1].time_stamp)
            break;
    }
    float factor = (t - rotations[idx].time_stamp) / (rotations[idx + 1].time_stamp - rotations[idx].time_stamp);
    return Quaternion::Slerp(rotations[idx].rotation, rotations[idx + 1].rotation, factor);
}

float3 Animation::interpolate_scale(float t) const
{
    if(scales.size() == 0) return { 1.0f, 1.0f, 1.0f };
    if(scales.size() == 1) return scales[0].scale;

    int idx = 0;
    for(; idx < scales.size() - 1; idx++)
    {
        if(t < scales[idx + 1].time_stamp)
            break;
    }
    float factor = (t - scales[idx].time_stamp) / (scales[idx + 1].time_stamp - scales[idx].time_stamp);
    return interpolate_float3(scales[idx].scale, scales[idx + 1].scale, factor);
}

Transform Animation::get_transform(float t) const
{
    t = extrapolate_time(t);
    float3 T = interpolate_translation(t);
    Quaternion R = interpolate_rotation(t);
    float3 S = interpolate_scale(t);
    return Transform::Compose(T, R, S);
}