#pragma once

#include <cuda_runtime.h>
#include "my_math.h"

enum class ACTION { UP, DOWN, LEFT, RIGHT, FRONT, BACK };

class Camera
{
public:
    float3 pos, front, up, right;
    float aspect, fov;
    float3 horizontal, vertical;

public:
    __host__ __device__ Ray get_ray(float x, float y) const
    {
        return Ray(pos, normalize(front + horizontal * (x - 0.5f) + vertical * (y - 0.5f)));
    }

    __host__ __device__ thrust::pair<float, float> get_xy(float3 dir) const
    {
        float cost_dir = dot(front, dir);
        if (cost_dir < 0.0f)
            return thrust::make_pair(-1.0f, -1.0f);
        float3 f = front * cost_dir;
        float3 v = (dir - f) * cost_dir / dot(f, dir);
        float x = dot(v, horizontal) / dot(horizontal, horizontal) + 0.5f;
        float y = dot(v, vertical) / dot(vertical, vertical) + 0.5f;
        return thrust::make_pair(x, y);
    }

    __host__ __device__ SquareMatrix<4> get_transform() const
    {
        return SquareMatrix<4>(
            right.x, right.y, right.z, -pos.x,
            up.x, up.y, up.z, -pos.y,
            front.x, front.y, front.z, -pos.z,
            0.0f, 0.0f, 0.0f, 1.0f);
    }

    __host__ __device__ SquareMatrix<4> get_inv_transform() const
    {
        auto m = Inverse(get_transform());
        return m.value();
    }

    // host function
    Camera(): pos({ 0.0f, 0.0f, 0.0f }), front({ 0.0f, 0.0f, 1.0f }), up({ 0.0f, 1.0f, 0.0f }),
        right({ 1.0f, 0.0f, 0.0f }), aspect(1.0f), fov(60.0f)
    {
        float nh = 2.0f * tan(radians(fov * 0.5f));
        horizontal = right * nh;
        vertical = up * nh / aspect;
    }

    Camera(SquareMatrix<4> transform, float _aspect = 1.0f, float _fov = 60.0f): aspect(_aspect), fov(_fov)
    {
        pos = make_float3(transform * make_float4(0.0f, 0.0f, 0.0f, 1.0f));
        front = make_float3(transform * make_float4(0.0f, 0.0f, 1.0f, 0.0f));
        up = make_float3(transform * make_float4(0.0f, 1.0f, 0.0f, 0.0f));
        right = make_float3(transform * make_float4(1.0f, 0.0f, 0.0f, 0.0f));

        float nh = 2.0f * tan(radians(fov * 0.5f));
        horizontal = right * nh;
        vertical = up * nh / aspect;
    }

    Camera(float3 _pos, float3 _target, float3 _up = make_float3(0.0f, 1.0f, 0.0f), float _aspect = 1.0f, float _fov = 60.0f)
        : pos(_pos), aspect(_aspect), fov(_fov)
    {
        front = normalize(_target - _pos);
        right = normalize(cross(front, _up));
        up = normalize(cross(right, front));

        float nh = 2.0f * tan(radians(fov * 0.5f));
        horizontal = right * nh;
        vertical = up * nh / aspect;
    }

    void process_keyboard_input(ACTION act, float m)
    {
        switch (act)
        {
        case ACTION::UP: pos += up * m; break;
        case ACTION::DOWN: pos -= up * m; break;
        case ACTION::LEFT: pos -= right * m; break;
        case ACTION::RIGHT: pos += right * m; break;
        case ACTION::FRONT: pos += front * m; break;
        case ACTION::BACK: pos -= front * m; break;
        }
    }

    void process_mouse_input(float xoffset, float yoffset)
    {
        SquareMatrix<4> m = get_transform();
        m = m * RotateX(radians(-yoffset)) * RotateY(radians(xoffset));

        front = make_float3(m * make_float4(0.0f, 0.0f, 1.0f, 0.0f));
        up = make_float3(m * make_float4(0.0f, 1.0f, 0.0f, 0.0f));
        right = make_float3(m * make_float4(1.0f, 0.0f, 0.0f, 0.0f));
    }

    void process_scroll_input(float yoffset)
    {
        fov = clamp(fov - yoffset, 1.0f, 90.0f);
        float nh = 2.0f * tan(radians(fov * 0.5f));
        horizontal = right * nh;
        vertical = up * nh / aspect;
    }
};

class Camera_
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

    __host__ __device__ Ray get_ray(float x, float y)
    {
        return Ray(pos, normalize(lower_left_corner + horizontal * x + vertical * y));
    }

    // dir must be normalized
    __host__ __device__ thrust::pair<float, float> get_xy(float3 dir)
    {
        float3 forward = lower_left_corner + horizontal * 0.5f + vertical * 0.5f;
        float cost = dot(forward, dir);
        if (cost < 0.0f)
            return thrust::make_pair(-1.0f, -1.0f);
        float3 f = forward * cost / dot(forward, forward);
        float3 v = (dir - f) * cost / dot(f, dir);
        float x = dot(v, horizontal) / dot(horizontal, horizontal) + 0.5f;
        float y = dot(v, vertical) / dot(vertical, vertical) + 0.5f;
        return thrust::make_pair(x, y);
    }

    // host function
    Camera_(): Camera_(make_float3(0.0f, 0.0f, 1.0f), 1.0f, 1.0f) {}

    Camera_(float3 _t, float _r, float _asp): Camera_(_t, _r, 0.0f, 90.0f, _asp, 60.0f) {}

    Camera_(float3 _t, float _r, float _phi, float _theta, float _asp, float _fov = 60.0f)
        : target(_t), radius(_r), phi(_phi), theta(_theta), aspect(_asp), fov(_fov)
    {
        set_camera();
    }

    void set_from_matrix(SquareMatrix<4> mat, float _fov, float _asp)
    {
        float3 forward = make_float3(mat[2][0], mat[2][1], mat[2][2]);
        float3 _up = make_float3(mat[1][0], mat[1][1], mat[1][2]);
        float3 right = make_float3(mat[0][0], mat[0][1], mat[0][2]);

        pos = make_float3(mat[3][0], -mat[3][1], mat[3][2]);
        float3 nh = normalize(cross(forward, _up));
        float3 nv = normalize(cross(nh, forward));

        float height = std::tan(radians(_fov * 0.5f));
        float width = height * _asp;

        horizontal = nh * width;
        vertical = nv * height;
        lower_left_corner = forward - horizontal * 0.5f - vertical * 0.5f;

        fov = _fov;
        aspect = _asp;
        changed = true;
    }

    void set_camera()
    {
        float cos_theta = std::cos(radians(theta)), sin_theta = std::sin(radians(theta));
        float cos_phi = std::cos(radians(phi)), sin_phi = std::sin(radians(phi));

        float3 forward = make_float3(-cos_theta * cos_phi, -sin_phi, -sin_theta * cos_phi);
        pos = target - forward * radius;
        float3 nh = normalize(cross(forward, up));
        float3 nv = normalize(cross(nh, forward));
        float height = std::tan(radians(fov * 0.5f));
        float width = height * aspect;
        horizontal = nh * width;
        vertical = nv * height;
        lower_left_corner = forward - horizontal * 0.5f - vertical * 0.5f;
        changed = true;
    }

    void process_keyboard_input(ACTION act, float deltaTime)
    {
        float m = keyboard_speed * deltaTime;
        switch (act)
        {
        case ACTION::UP: target = target + normalize(vertical) * m; break;
        case ACTION::DOWN: target = target - normalize(vertical) * m; break;
        case ACTION::LEFT: target = target - normalize(horizontal) * m; break;
        case ACTION::RIGHT: target = target + normalize(horizontal) * m; break;
        }

        set_camera();
    }

    void process_mouse_input(float xoffset, float yoffset)
    {
        theta += xoffset * mouse_sensitivity;
        phi = clamp(phi - yoffset * mouse_sensitivity, -89.0f, 89.0f);

        set_camera();
    }

    void process_scroll_input(float yoffset)
    {
        fov = clamp(fov - yoffset, 1.0f, 120.0f);

        set_camera();
    }
};