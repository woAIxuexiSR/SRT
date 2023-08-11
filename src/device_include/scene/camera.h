#pragma once

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
    enum class Type { Orbit, FPS };

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

    __device__ float3 to_local(float3 dir) const
    {
        return make_float3(dot(dir, x), dot(dir, y), dot(dir, z));
    }

    /* host functions */

    CameraController(const Transform& t, float _r, Type _t = Type::Orbit)
        : type(_t), radius(_r)
    {
        // camera to world transform
        pos = make_float3(t[0][3], t[1][3], t[2][3]);
        x = normalize(make_float3(t[0][0], t[1][0], t[2][0]));
        y = normalize(make_float3(t[0][1], t[1][1], t[2][1]));
        z = normalize(make_float3(t[0][2], t[1][2], t[2][2]));
        target = pos + z * radius;

        phi = Degrees(atan2(z.z, z.x)); // phi is in [-180, 180], angle between projected z and (1, 0, 0)
        theta = Degrees(acos(z.y));     // theta is in [0, 180], angle between z and (0, 1, 0)
    }

    CameraController() : CameraController(Transform(), 1.0f) {}

    void reset()
    {
        float cos_theta = cos(Radians(theta)), sin_theta = sin(Radians(theta));
        float cos_phi = cos(Radians(phi)), sin_phi = sin(Radians(phi));
        z = make_float3(sin_theta * cos_phi, cos_theta, sin_theta * sin_phi);
        x = make_float3(sin_phi, 0.0f, -cos_phi);                               // cross((0, 1, 0), z)
        y = make_float3(-cos_theta * cos_phi, sin_theta, -cos_theta * sin_phi); // cross(z, x)

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
        case CameraMovement::LEFT: pos += x * m; target += x * m; break;
        case CameraMovement::RIGHT: pos -= x * m; target -= x * m; break;
        case CameraMovement::FORWARD: pos += z * m; target += z * m; break;
        case CameraMovement::BACKWARD: pos -= z * m; target -= z * m; break;
        }
    }

    void process_mouse_input(float xoffset, float yoffset)
    {
        theta = clamp(theta + yoffset, 1.0f, 179.0f);
        phi += xoffset;
        reset();
    }
};

/* face to z axis, film plane is at z = -1 */
class Camera
{
public:
    enum class Type { Perspective, Orthographic, ThinLens, Environment };

    CameraController controller;
    Type type;
    float aspect, fov;          // fov : in degree, vertical
    float aperture, focal;      // thin lens

    float nh, nw;               // height and width of film plane
    bool moved{ false };        // moved flag

public:
    __device__ Ray get_ray(float x, float y, RandomGenerator& rng) const
    {
        float2 p = make_float2((x - 0.5f) * nw, (y - 0.5f) * nh);

        switch (type)
        {
        case Type::Perspective:             // automatically flip y
        {
            Ray ray(make_float3(0.0f), normalize(make_float3(p, 1.0f)));
            return controller.to_world(ray);
        }
        case Type::Orthographic:            // approximate size
        {
            Ray ray(make_float3(p * controller.radius, 1.0f), make_float3(0.0f, 0.0f, 1.0f));
            return controller.to_world(ray);
        }
        case Type::ThinLens:
        {
            float2 uv = uniform_sample_disk(rng.random_float2());
            float3 pos = make_float3(uv * aperture, 0.0f);
            float3 target = make_float3(p, 1.0f) * focal;
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

    // only support perspective, the ray must pass through the camera center in reverse direction
    __device__ float2 get_xy(float3 dir) const
    {
        dir = controller.to_local(-dir);
        float2 p = make_float2(dir.x, dir.y) / dir.z;
        return make_float2(p.x / nw + 0.5f, p.y / nh + 0.5f);
    }

    /* host functions */

    Camera() : controller(), type(Type::Perspective), aspect(1.0f), fov(60.0f), aperture(0.0f), focal(1.0f)
    {
        reset();
    }
    Camera(Type _t, float _aspect, float _fov)
        : controller(), type(_t), aspect(_aspect), fov(_fov), aperture(0.0f), focal(1.0f)
    {
        reset();
    }

    void set_moved(bool _moved) { moved = _moved; }
    void set_controller(Transform t, float r) { controller = CameraController(t, r); }

    void reset()
    {
        nh = 2.0f * tan(Radians(fov * 0.5f));
        nw = nh * aspect;
        moved = true;
    }

    void process_scroll_input(float yoffset)
    {
        fov = clamp(fov - yoffset, 1.0f, 90.0f);
        reset();
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
};