# pragma once

#include "definition.h"
#include "my_math.h"

struct KeyTranslation
{
    float time_stamp;
    float3 translation;
};

struct KeyRotation
{
    float time_stamp;
    Quaternion rotation;
};

struct KeyScale
{
    float time_stamp;
    float3 scale;
};

class Animation
{
public:
    enum class Interpolation { Linear, Cubic };
    enum class Extrapolation { Clamp, Repeat, Mirror };

    Interpolation itype { Interpolation::Linear };
    Extrapolation etype { Extrapolation::Repeat };
    string name;
    float duration { 0.0f };    // in seconds
    vector<KeyTranslation> translations;
    vector<KeyRotation> rotations;
    vector<KeyScale> scales;

public:
    Animation() {}

    /* helper functions */

    float3 interpolate_float3(float3 a, float3 b, float factor) const;
    float extrapolate_time(float t) const;

    float3 interpolate_translation(float t) const;
    Quaternion interpolate_rotation(float t) const;
    float3 interpolate_scale(float t) const;

    /* useful functions */
    
    Transform get_transform(float t) const;
};