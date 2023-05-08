#pragma once

#include "definition.h"
#include "helper_optix.h"
#include "helper_cuda.h"
#include "my_math.h"

#include "scene.h"

#include "scene/camera.h"
#include "scene/material.h"
#include "scene/light.h"


class OptixRayTracer
{
protected:
    // context
    CUstream stream;
    OptixDeviceContext context;

    // module
    vector<ModuleProgramGroup> module_pgs;
    OptixPipelineCompileOptions pipeline_compile_options;
    vector<OptixPipeline> pipelines;

    // scene
    shared_ptr<Scene> scene;
    vector<GPUMemory<float3> > vertex_buffer;
    vector<GPUMemory<uint3> > index_buffer;
    vector<GPUMemory<float3> > normal_buffer;
    vector<GPUMemory<float2> > texcoord_buffer;

    OptixTraversableHandle traversable;
    GPUMemory<unsigned char> as_buffer;

    // texture
    vector<cudaArray_t> texture_arrays;
    vector<cudaTextureObject_t> texture_objects;

    // sbt
    vector<OptixShaderBindingTable> sbts;
    vector<GPUMemory<RaygenSBTRecord> > raygen_sbt;
    vector<GPUMemory<MissSBTRecord> > miss_sbt;
    vector<GPUMemory<HitgroupSBTRecord> > hitgroup_sbt;

    // light
    Light light;
    
    GPUMemory<DiffuseAreaLight> diffuse_area_light_buffer;
    vector<GPUMemory<float> > light_area_buffer;
    vector<int> meshid_to_lightid;

    GPUMemory<InfiniteLight> infinite_light_buffer;
    cudaArray_t environment_map_array;
    cudaTextureObject_t environment_map;

private:
    void init_optix();
    void create_context();
    void create_module(const string& ptx);
    void create_pipeline(const vector<string>& ptxs);
    void build_as();
    void create_textures();
    void create_environment_map();
    void create_light();
    void build_sbt();

public:
    OptixRayTracer(const vector<string>& _ptxfiles, shared_ptr<Scene> _scene);

    OptixTraversableHandle get_traversable() const { return traversable; }
    Light get_light() const { return light; }


    template<class T>
    void trace(int num, int idx, const GPUMemory<T>& params)
    {
        OPTIX_CHECK(optixLaunch(
            pipelines[idx],
            stream,
            (CUdeviceptr)params.data(),
            params.size() * sizeof(T),
            &sbts[idx],
            num, 1, 1
        ));
    }
};