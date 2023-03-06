#pragma once

#include <cuda_runtime.h>
#include <iostream>
#include <filesystem>
#include <vector>
#include <set>
#include <map>

#include "srt_math.h"
#include "definition.h"
#include "scene/material.h"


class TriangleMesh
{
public:
    std::vector<float3> vertices;
    std::vector<uint3> indices;
    std::vector<float3> normals;
    std::vector<float2> texcoords;

    int materialId{ -1 };
    int textureId{ -1 };
};


class MaterialParameter
{
public:
    MaterialType type;
    float3 color;

    // glass material parameters
    float ior;

    // disney material parameters
    float metallic{ 0.0f };
    float roughness{ 0.5f };
};


class Texture
{
public:
    unsigned* pixels;
    uint2 resolution;

    Texture(): pixels(nullptr), resolution({ 0, 0 }) {}
    ~Texture() { if (pixels) delete[] pixels; }
};


class Model
{
public:
    std::vector<TriangleMesh*> meshes;
    std::vector<MaterialParameter> material_params;
    std::vector<Texture*> textures;

public:
    Model() {}
    Model(const std::string& objPath);
    ~Model();

    void loadObj(const std::string& objPath);
    int loadTexture(std::map<std::string, int>& knownTextures, const std::string& TextureName, const std::string& modelDir);
};