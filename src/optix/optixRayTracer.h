#pragma once

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "helper_optix.h"
#include "helper_cuda.h"
#include "scene.h"
#include "film.h"
#include "scene/camera.h"
#include "launchParams/LaunchParams.h"
#include "scene/material.h"
#include "scene/light.h"

class ModuleProgramGroup
{
public:
    OptixProgramGroup raygenPG;
    OptixProgramGroup missPGs[2];
    OptixProgramGroup hitgroupPGs[2];
};

class OptixRayTracer
{
protected:
    std::vector<std::string> modulePTXs;

    CUstream stream;
    OptixDeviceContext optixContext;
    std::vector<ModuleProgramGroup> modulePGs;
    std::vector<OptixPipeline> pipelines;

    const Model* model;
    std::vector<GPUMemory<float3> > vertexBuffer;
    std::vector<GPUMemory<uint3> > indexBuffer;
    std::vector<GPUMemory<float2> > texcoordBuffer;
    std::vector<GPUMemory<float3> > normalBuffer;

    Light light;
    GPUMemory<float3> lightVertexBuffer;
    GPUMemory<float3> lightNormalBuffer;
    GPUMemory<uint3> lightIndexBuffer;
    GPUMemory<float> lightAccumAreaBuffer;
    GPUMemory<float3> lightEmissionBuffer;

    std::vector<cudaArray_t> textureArrays;
    std::vector<cudaTextureObject_t> textureObjects;

    std::vector<OptixShaderBindingTable> sbts;
    std::vector<GPUMemory<RaygenSBTRecord> > raygenRecordsBuffer;
    std::vector<GPUMemory<MissSBTRecord> > missRecordsBuffer;
    std::vector<GPUMemory<HitgroupSBTRecord> > hitgroupRecordsBuffer;

    OptixTraversableHandle traversable;
    GPUMemory<unsigned char> asBuffer;      // preserve

    int width, height;

    void initOptix();

    void createContext();
    void createModule(const std::string& ptx, OptixPipelineCompileOptions& pipelineCompileOptions, std::vector<OptixProgramGroup>& programGroups);
    void createPipelines();

    void buildAccel();
    void createTextures();
    void buildSBT();

    void generateLight();

public:
    OptixRayTracer(const std::vector<std::string>& _ptxfiles, const Model* _model, int _w, int _h);

    virtual void render(std::shared_ptr<Camera> camera, std::shared_ptr<Film> film) = 0;
};