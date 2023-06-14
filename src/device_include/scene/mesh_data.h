#pragma once

#include <cuda_runtime.h>
#include "material.h"

class MeshData
{
public:
    float3* vertices { nullptr };
    uint3* indices { nullptr };
    float3* normals { nullptr };
    float3* tangents { nullptr };
    float2* texcoords { nullptr };

    Material* material { nullptr };
};