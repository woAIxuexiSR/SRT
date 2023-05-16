#pragma once

#include <cuda_runtime.h>
#include "my_math.h"

/*
right hand coordinate system
    - z axis : front
    - x axis : left
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
    float theta, phi;   // different meaning for different type
    float radius;       // only for orbit

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
    CameraController(CameraController::Type _t, const Transform& t, float _radius = 1.0f)
        : type(_t), radius(_radius)
    {
        pos = make_float3(t[0][3], t[1][3], t[2][3]);
        x = make_float3(t[0][0], t[1][0], t[2][0]);
        y = make_float3(t[0][1], t[1][1], t[2][1]);
        z = make_float3(t[0][2], t[1][2], t[2][2]);
        target = pos + z * radius;

        switch (type)
        {
        case Type::Orbit:
        {
            theta = acos(-z.y) * 180.0f / M_PI;
            phi = atan2(-z.z, -z.x) * 180.0f / M_PI;
            break;
        }
        case Type::FPS:
        {
            theta = acos(z.y) * 180.0f / M_PI;
            phi = atan2(z.z, z.x) * 180.0f / M_PI;
            break;
        }
        default: break;
        }
    }

    CameraController() : CameraController(Type::None, Transform()) {}

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
        switch (type)
        {
        case Type::Orbit:
        {
            theta = clamp(theta + yoffset, 1.0f, 179.0f);
            phi += xoffset;

            float cos_theta = cos(radians(theta)), sin_theta = sin(radians(theta));
            float cos_phi = cos(radians(phi)), sin_phi = sin(radians(phi));
            z = 0.0f - make_float3(cos_phi * sin_theta, cos_theta, sin_phi * sin_theta);
            x = normalize(cross(make_float3(0.0f, 1.0f, 0.0f), z));
            y = normalize(cross(z, x));
            pos = target - z * radius;

            break;
        }
        case Type::FPS:
        {
            theta = clamp(theta - yoffset, 1.0f, 179.0f);
            phi += xoffset;

            float cos_theta = cos(radians(theta)), sin_theta = sin(radians(theta));
            float cos_phi = cos(radians(phi)), sin_phi = sin(radians(phi));
            z = make_float3(cos_phi * sin_theta, cos_theta, sin_phi * sin_theta);
            x = normalize(cross(make_float3(0.0f, 1.0f, 0.0f), z));
            y = normalize(cross(z, x));
            target = pos + z * radius;

            break;
        }
        default:
            break;
        }
    }
};


// film plane is at z = -1
class Camera
{
public:
    enum class Mode { Perspective, Orthographic, ThinLens, Environment };

    CameraController controller;
    Mode mode;
    float aspect;
    float fov, nh, nw;  // for perspective, thin lens
    float scale;        // for orthographic
    float focal, aperture;  // for thin lens

public:
    __device__ Ray get_ray(float x, float y, RandomGenerator& rng) const
    {
        switch (mode)
        {
        case Mode::Perspective:
        {
            float3 dir = normalize(make_float3((x - 0.5f) * nw, (y - 0.5f) * nh, 1.0f));
            return controller.to_world(Ray(make_float3(0.0f), dir));
        }
        case Mode::Orthographic:
        {
            float3 pos = make_float3((x - 0.5f) * scale * aspect, (y - 0.5f) * scale, 0.0f);
            return controller.to_world(Ray(pos, make_float3(0.0f, 0.0f, 1.0f)));
        }
        case Mode::ThinLens:
        {
            float2 uv = uniform_sample_disk(rng.random_float2());
            float3 pos = make_float3(uv * aperture, 0.0f);
            float3 target = make_float3((x - 0.5f) * nw, (y - 0.5f) * nh, 1.0f) * focal;
            return controller.to_world(Ray(pos, normalize(target - pos)));
        }
        case Mode::Environment:
        {
            float2 spherical = make_float2(x * 2.0f * M_PI, y * M_PI);
            float3 dir = spherical_to_dir(spherical);
            return controller.to_world(Ray(make_float3(0.0f), dir));
        }
        default:
            return Ray();
        }
    }

    __device__ thrust::pair<float, float> get_xy(float3 dir) const
    {
        // float cost_dir = dot(front, dir);
        // if (cost_dir < 0.0f)
        //     return thrust::make_pair(-1.0f, -1.0f);
        // float3 f = front * cost_dir;
        // float3 v = (dir - f) * cost_dir / dot(f, dir);
        // float x = dot(v, horizontal) / dot(horizontal, horizontal) + 0.5f;
        // float y = dot(v, vertical) / dot(vertical, vertical) + 0.5f;
        // return thrust::make_pair(x, y);
        return thrust::make_pair(0.0f, 0.0f);
    }

    // host function
    Camera(Mode _m, float _aspect, float _fov = 60.0f)
        : mode(_m), aspect(_aspect), fov(_fov)
    {
        nw = 2.0f * tan(radians(fov * 0.5f));
        nh = nw / aspect;
    }

    Camera() : Camera(Mode::Perspective, 1.0f) {}

    void set_controller(CameraController::Type _t, const Transform& camera_to_world, float _radius = 1.0f)
    {
        controller = CameraController(_t, camera_to_world, _radius);
    }

    void set_orhtographic(float _scale)
    {
        scale = _scale;
    }

    void set_thin_lens(float _focal, float _aperture)
    {
        focal = _focal;
        aperture = _aperture;
    }


    void process_keyboard_input(CameraMovement movement, float m)
    {
        controller.process_keyboard_input(movement, m);
    }

    void process_mouse_input(float xoffset, float yoffset)
    {
        controller.process_mouse_input(xoffset, yoffset);
    }

    void process_scroll_input(float yoffset)
    {
        fov = clamp(fov - yoffset, 1.0f, 90.0f);
        nw = 2.0f * tan(radians(fov * 0.5f));
        nh = nw / aspect;
    }
};