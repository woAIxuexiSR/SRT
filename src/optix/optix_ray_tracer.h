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
    // scene
    shared_ptr<Scene> scene;

    // context
    CUstream stream;
    OptixDeviceContext context;

    // module
    vector<ModuleProgramGroup> module_pgs;
    OptixPipelineCompileOptions pipeline_compile_options;
    vector<OptixPipeline> pipelines;

    // traversable
    OptixTraversableHandle traversable;
    GPUMemory<unsigned char> as_buffer;

    // sbt
    vector<OptixShaderBindingTable> sbts;
    vector<GPUMemory<RaygenSBTRecord> > raygen_sbt;
    vector<GPUMemory<MissSBTRecord> > miss_sbt;
    vector<GPUMemory<HitgroupSBTRecord> > hitgroup_sbt;

private:
    void init_optix();
    void create_context();
    void create_module(const string& ptx);
    void create_pipeline(const vector<string>& ptxs);
    void build_as();
    void build_sbt();

public:
    OptixRayTracer(const vector<string>& _ptxfiles, shared_ptr<Scene> _scene);
    OptixTraversableHandle get_traversable() const { return traversable; }


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