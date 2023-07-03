#pragma once

#include "my_math.h"
#include "gmaterial.h"

class GTriangleMesh
{
public:
    float3* vertices{ nullptr };
    uint3* indices{ nullptr };
    float3* normals{ nullptr };
    float3* tangents{ nullptr };
    float2* texcoords{ nullptr };

    GMaterial* material{ nullptr };

public:
    GTriangleMesh() {}
    GTriangleMesh(float3* _v, uint3* _i, float3* _n, float3* _t, float2* _tc, GMaterial* _m)
        : vertices(_v), indices(_i), normals(_n), tangents(_t), texcoords(_tc), material(_m) {}
};

class GInstance
{
public:
    Transform* transform{ nullptr };
    GTriangleMesh* mesh{ nullptr };
};