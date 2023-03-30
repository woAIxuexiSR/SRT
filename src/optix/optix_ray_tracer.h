#pragma once

#include "definition.h"
#include "helper_optix.h"
#include "helper_cuda.h"

#include "scene.h"
#include "film.h"

#include "launch_params/launch_params.h"

#include "scene/camera.h"
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
    vector<string> modulePTXs;

    CUstream stream;
    OptixDeviceContext optixContext;
    vector<ModuleProgramGroup> modulePGs;
    vector<OptixPipeline> pipelines;

    const Scene* scene;
    vector<GPUMemory<float3> > vertexBuffer;
    vector<GPUMemory<uint3> > indexBuffer;
    vector<GPUMemory<float2> > texcoordBuffer;
    vector<GPUMemory<float3> > normalBuffer;

    Light light;
    GPUMemory<float3> lightVertexBuffer;
    GPUMemory<float3> lightNormalBuffer;
    GPUMemory<uint3> lightIndexBuffer;
    GPUMemory<float> lightAccumAreaBuffer;
    GPUMemory<float3> lightEmissionBuffer;

    vector<cudaArray_t> textureArrays;
    vector<cudaTextureObject_t> textureObjects;

    vector<OptixShaderBindingTable> sbts;
    vector<GPUMemory<RaygenSBTRecord> > raygenRecordsBuffer;
    vector<GPUMemory<MissSBTRecord> > missRecordsBuffer;
    vector<GPUMemory<HitgroupSBTRecord> > hitgroupRecordsBuffer;

    OptixTraversableHandle traversable;
    GPUMemory<unsigned char> asBuffer;      // preserve


    void initOptix();

    void createContext();
    void createModule(const string& ptx, OptixPipelineCompileOptions& pipelineCompileOptions, vector<OptixProgramGroup>& programGroups);
    void createPipelines();

    void buildAccel();
    void createTextures();
    void buildSBT();

    void generateLight();

public:
    int spp {4};
    float3 background{ 0.0f, 0.0f, 0.0f };

public:
    OptixRayTracer(const vector<string>& _ptxfiles, const Scene* _scene);

    void set_spp(int _spp) { spp = _spp; }
    void set_background(float3 _background) { background = _background; }
    virtual void render(shared_ptr<Camera> camera, shared_ptr<Film> film) = 0;
};