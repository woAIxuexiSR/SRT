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

    int materialId;
    float3 emittance;
    int textureId{ -1 };

    // material type, index

public:
    void addVertices(const std::vector<float3>& _v, const std::vector<uint3>& _i);
    void addCube(const float3& center, const float3& size);
    void addSquare_XZ(const float3& center, const float2& size);
    void addSquare_XY(const float3& center, const float2& size);
    void addSquare_YZ(const float3& center, const float2& size);

    // void setColor(const float3& _c) { diffuse = _c; }
    void setEmittance(const float3& _e) { emittance = _e; }
};

class Texture
{
public:
    unsigned* pixels;
    uint2 resolution;

    Texture() : pixels(nullptr), resolution({ 0, 0 }) {}
    ~Texture() { if (pixels) delete[] pixels; }
};

class Model
{
public:
    std::vector<TriangleMesh*> meshes;
    std::vector<Texture*> textures;
    std::vector<DiffuseMaterial> diffuseMaterials;

public:
    Model() {}
    Model(const std::string& objPath);
    ~Model();

    void loadObj(const std::string& objPath);
    void loadCornellBox();
    int loadTexture(std::map<std::string, int>& knownTextures, const std::string& TextureName, const std::string& modelDir);
};