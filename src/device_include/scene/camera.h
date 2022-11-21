#pragma once

#include <cuda_runtime.h>
#include "srt_math.h"

enum class ACTION { UP, DOWN, LEFT, RIGHT };

class Camera
{
public:
    float3 target;
    float radius, phi, theta;
    float aspect, fov;

    float3 pos, horizontal, vertical, lower_left_corner;
    bool changed;

    static constexpr float3 up = { 0.0f, 1.0f, 0.0f };
    static constexpr float keyboard_speed = 2.5f;
    static constexpr float mouse_sensitivity = 0.2f;

    __host__ __device__ Ray getRay(float x, float y)
    {
        return Ray(pos, normalize(lower_left_corner + horizontal * x + vertical * y));
    }

    // host function
    Camera() : Camera(make_float3(0.0f, 0.0f, 1.0f), 1.0f, 1.0f) {}

    Camera(float3 _t, float _r, float _asp) : Camera(_t, _r, 0.0f, 90.0f, _asp, 60.0f) {}

    Camera(float3 _t, float _r, float _phi, float _theta, float _asp, float _fov = 60.0f)
        : target(_t), radius(_r), phi(_phi), theta(_theta), aspect(_asp), fov(_fov)
    {
        SetCamera();
    }

    void SetCamera()
    {
        float cos_theta = std::cos(Radians(theta)), sin_theta = std::sin(Radians(theta));
        float cos_phi = std::cos(Radians(phi)), sin_phi = std::sin(Radians(phi));

        float3 forward = make_float3(-cos_theta * cos_phi, -sin_phi, -sin_theta * cos_phi);
        pos = target - forward * radius;
        float3 nh = normalize(cross(forward, up));
        float3 nv = normalize(cross(nh, forward));
        float height = std::tan(Radians(fov * 0.5f));
        float width = height * aspect;
        horizontal = nh * width;
        vertical = nv * height;
        lower_left_corner = forward - horizontal * 0.5f - vertical * 0.5f;
        changed = true;
    }

    void ProcessKeyboardInput(ACTION act, float deltaTime)
    {
        float m = keyboard_speed * deltaTime;
        switch (act)
        {
        case ACTION::UP: target = target + normalize(vertical) * m; break;
        case ACTION::DOWN: target = target - normalize(vertical) * m; break;
        case ACTION::LEFT: target = target - normalize(horizontal) * m; break;
        case ACTION::RIGHT: target = target + normalize(horizontal) * m; break;
        }

        SetCamera();
    }

    void ProcessMouseInput(float xoffset, float yoffset)
    {
        theta += xoffset * mouse_sensitivity;
        phi = clamp(phi - yoffset * mouse_sensitivity, -89.0f, 89.0f);

        SetCamera();
    }

    void ProcessScrollInput(float yoffset)
    {
        fov = clamp(fov - yoffset, 1.0f, 120.0f);

        SetCamera();
    }
};