#pragma once

#include "scene.h"

#include "helper_optix.h"
#include "helper_cuda.h"

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
    vector<CUdeviceptr> d_vertices;
    vector<uint32_t> gas_input_flags;
    vector<OptixBuildInput> gas_inputs;
    vector<GPUMemory<unsigned char> > gas_buffer;
    vector<OptixTraversableHandle> gas_traversable;

    vector<OptixInstance> ias_instances;
    GPUMemory<OptixInstance> ias_instances_buffer;
    OptixBuildInput ias_input;
    GPUMemory<unsigned char> ias_buffer;
    OptixTraversableHandle ias_traversable;

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
    OptixTraversableHandle build_as_from_input(const vector<OptixBuildInput>& inputs, GPUMemory<unsigned char>& as_buffer, bool compact, bool update);
    void build_gas();
    void build_ias();
    void build_sbt();

public:
    OptixRayTracer(const vector<string>& _ptxfiles, shared_ptr<Scene> _scene);
    OptixTraversableHandle get_traversable() const { return ias_traversable; }
    void update_as();

    template<class T>
    void trace(int num, int idx, T* params)
    {
        OPTIX_CHECK(optixLaunch(
            pipelines[idx],
            stream,
            (CUdeviceptr)params,
            sizeof(T),
            &sbts[idx],
            num, 1, 1
        ));
    }
};