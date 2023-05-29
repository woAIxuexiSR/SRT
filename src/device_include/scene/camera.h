#pragma once

#include <cuda_runtime.h>
#include "my_math.h"

/*
left hand coordinate system
    - z axis : front
    - x axis : right
    - y axis : up
*/

enum class CameraMovement { UP, DOWN, LEFT, RIGHT, FORWARD, BACKWARD };

class CameraController
{
public:
    enum class Type { None, Orbit, FPS };

    Type type;
    float3 pos, target;
    float3 z, x, y;
    float phi, theta;   // z axis's spherical coordinate, in world space, in degree
    float radius;       // distance between pos and target

public:
    __device__ Ray to_world(Ray ray) const
    {
        float3 p = ray.pos.x * x + ray.pos.y * y + ray.pos.z * z + pos;
        float3 d = ray.dir.x * x + ray.dir.y * y + ray.dir.z * z;
        return Ray(p, d);
    }

    __device__ Ray to_local(Ray ray) const
    {
        float3 p = ray.pos - pos;
        p = make_float3(dot(p, x), dot(p, y), dot(p, z));
        float3 d = make_float3(dot(ray.dir, x), dot(ray.dir, y), dot(ray.dir, z));
        return Ray(p, d);
    }

    // host function
    CameraController(const Transform& t, float _r, Type _t = Type::Orbit)
        : type(_t), radius(_r)
    {
        pos = make_float3(t[0][3], t[1][3], t[2][3]);
        x = make_float3(t[0][0], t[1][0], t[2][0]);
        y = make_float3(t[0][1], t[1][1], t[2][1]);
        z = make_float3(t[0][2], t[1][2], t[2][2]);
        target = pos + z * radius;

        phi = Degrees(atan2(z.z, z.x));
        theta = Degrees(acos(z.y));
    }

    CameraController() : CameraController(Transform(), 1.0f) {}

    void reset_from_angle()
    {
        float cos_theta = cos(Radians(theta)), sin_theta = sin(Radians(theta));
        float cos_phi = cos(Radians(phi)), sin_phi = sin(Radians(phi));
        z = make_float3(cos_phi * sin_theta, cos_theta, sin_phi * sin_theta);
        x = normalize(cross(z, make_float3(0.0f, 1.0f, 0.0f)));
        y = normalize(cross(x, z));

        if (type == Type::Orbit)
            pos = target - z * radius;
        else
            target = pos + z * radius;
    }

    void process_keyboard_input(CameraMovement movement, float m)
    {
        switch (movement)
        {
        case CameraMovement::UP: pos += y * m; target += y * m; break;
        case CameraMovement::DOWN: pos -= y * m; target -= y * m; break;
        case CameraMovement::LEFT: pos -= x * m; target -= x * m; break;
        case CameraMovement::RIGHT: pos += x * m; target += x * m; break;
        case CameraMovement::FORWARD: pos += z * m; target += z * m; break;
        case CameraMovement::BACKWARD: pos -= z * m; target -= z * m; break;
        }
    }

    void process_mouse_input(float xoffset, float yoffset)
    {
        if (type == Type::None) return;

        theta = clamp(theta + yoffset, 1.0f, 179.0f);
        phi += xoffset;

        reset_from_angle();
    }
};


// film plane is at z = -1
class Camera
{
public:
    enum class Type { Perspective, Orthographic, ThinLens, Environment };

    CameraController controller;
    Type type;
    float aspect;
    float fov, nh, nw;      // for perspective, thin lens. for orthographic, scale = nw * radius
    float focal{ 1.0f }, aperture{ 0.0f };  // for thin lens

    bool moved{ false };

public:
    __device__ Ray get_ray(float x, float y, RandomGenerator& rng) const
    {
        switch (type)
        {
        case Type::Perspective:
        {
            float3 dir = normalize(make_float3((x - 0.5f) * nw, (y - 0.5f) * nh, 1.0f));
            return controller.to_world(Ray(make_float3(0.0f), dir));
        }
        case Type::Orthographic:
        {
            float radius = controller.radius;
            float3 pos = make_float3((x - 0.5f) * nw * radius, (y - 0.5f) * nh * radius, 0.0f);
            return controller.to_world(Ray(pos, make_float3(0.0f, 0.0f, 1.0f)));
        }
        case Type::ThinLens:
        {
            float2 uv = uniform_sample_disk(rng.random_float2());
            float3 pos = make_float3(uv * aperture, 0.0f);
            float3 target = make_float3((x - 0.5f) * nw, (y - 0.5f) * nh, 1.0f) * focal;
            return controller.to_world(Ray(pos, normalize(target - pos)));
        }
        case Type::Environment:
        {
            float2 phi_theta = make_float2((x - 0.5f) * 2.0f * M_PI, y * M_PI);
            float3 math_space_dir = spherical_uv_to_cartesian(phi_theta);
            float3 dir = make_float3(-math_space_dir.x, math_space_dir.z, math_space_dir.y);
            return controller.to_world(Ray(make_float3(0.0f), dir));
        }
        default:
            return Ray();
        }
    }

    __device__ float2 get_xy(float3 dir) const
    {
        // float cost_dir = dot(front, dir);
        // if (cost_dir < 0.0f)
        //     return thrust::make_pair(-1.0f, -1.0f);
        // float3 f = front * cost_dir;
        // float3 v = (dir - f) * cost_dir / dot(f, dir);
        // float x = dot(v, horizontal) / dot(horizontal, horizontal) + 0.5f;
        // float y = dot(v, vertical) / dot(vertical, vertical) + 0.5f;
        // return thrust::make_pair(x, y);
        return { 0.0f, 0.0f };
    }

    // host function
    Camera() {}

    void set_type(Type _t) { type = _t; }
    void set_aspect_fov(float _aspect, float _fov)
    {
        aspect = _aspect;
        fov = _fov;
        nw = 2.0f * tan(Radians(fov * 0.5f));
        nh = nw / aspect;
    }
    void set_controller(const Transform& camera_to_world, float _r,
        CameraController::Type _t = CameraController::Type::Orbit)
    {
        controller = CameraController(camera_to_world, _r, _t);
    }
    void set_focal_aperture(float _focal, float _aperture)
    {
        focal = _focal;
        aperture = _aperture;
    }


    void process_keyboard_input(CameraMovement movement, float m)
    {
        controller.process_keyboard_input(movement, m);
        moved = true;
    }

    void process_mouse_input(float xoffset, float yoffset)
    {
        controller.process_mouse_input(xoffset, yoffset);
        moved = true;
    }

    void process_scroll_input(float yoffset)
    {
        fov = clamp(fov - yoffset, 1.0f, 90.0f);
        nw = 2.0f * tan(Radians(fov * 0.5f));
        nh = nw / aspect;
        moved = true;
    }
};