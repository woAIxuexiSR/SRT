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
};