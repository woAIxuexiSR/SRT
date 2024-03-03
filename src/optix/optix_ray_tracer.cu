#include "optix_ray_tracer.h"
#include <optix_function_table_definition.h>

string get_ptx_from_file(const string& filename)
{
    std::ifstream fs(filename, std::ios::in);
    if (!fs.is_open())
    {
        cout << "ERROR::Failed to open file: " << filename << endl;
        exit(-1);
    }

    std::stringstream ss;
    ss << fs.rdbuf();
    fs.close();

    return ss.str();
}

void OptixRayTracer::init_optix()
{
    checkCudaErrors(cudaFree(0));

    int device_num;
    checkCudaErrors(cudaGetDeviceCount(&device_num));
    if (device_num == 0)
    {
        cout << "ERROR::No CUDA capable devices found!" << endl;
        exit(-1);
    }
    cout << "Found " << device_num << " CUDA devices" << endl;

    OPTIX_CHECK(optixInit());
    cout << "Successfully initialized optix" << endl;
}

void OptixRayTracer::create_context()
{
    const int device_id = 0;
    checkCudaErrors(cudaSetDevice(device_id));
    checkCudaErrors(cudaStreamCreate(&stream));

    cudaDeviceProp device_prop;
    checkCudaErrors(cudaGetDeviceProperties(&device_prop, device_id));
    cout << "Running on device: " << device_prop.name << endl;

    CUcontext cuda_context;
    CUresult cu_res = cuCtxGetCurrent(&cuda_context);
    if (cu_res != CUDA_SUCCESS)
    {
        cout << "ERROR::Failed to get current CUDA context" << endl;
        exit(-1);
    }

    OPTIX_CHECK(optixDeviceContextCreate(cuda_context, 0, &context));
}

void OptixRayTracer::create_module(const string& ptx)
{
    OptixModule module;

    OptixModuleCompileOptions module_compile_options = {};
    module_compile_options.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_DEFAULT;

#if (OPTIX_VERSION >= 70700)
#define OPTIX_MODULE_CREATE optixModuleCreate
#else
#define OPTIX_MODULE_CREATE optixModuleCreateFromPTX
#endif

    // create module
    char log[2048];
    size_t log_size = sizeof(log);
    OPTIX_CHECK(OPTIX_MODULE_CREATE(
        context,
        &module_compile_options,
        &pipeline_compile_options,
        ptx.c_str(),
        ptx.size(),
        log,
        &log_size,
        &module
    ));
    if (log_size > 1)
        cout << "Optix module log: " << log << endl;

    ModuleProgramGroup module_pg;

    // create raygen program
    {
        OptixProgramGroupOptions pg_options = {};
        OptixProgramGroupDesc pg_desc = {};

        pg_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        pg_desc.raygen.module = module;
        pg_desc.raygen.entryFunctionName = "__raygen__";

        char log[2048];
        size_t log_size = sizeof(log);
        OPTIX_CHECK(optixProgramGroupCreate(
            context,
            &pg_desc,
            1,
            &pg_options,
            log,
            &log_size,
            &module_pg.raygenPG
        ));
        if (log_size > 1)
            cout << "Optix raygen program log: " << log << endl;
    }

    // create miss program
    {
        string name[2] = { "__miss__radiance", "__miss__shadow" };
        for (int i = 0; i < 2; i++)
        {
            OptixProgramGroupOptions pg_options = {};
            OptixProgramGroupDesc pg_desc = {};

            pg_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
            pg_desc.miss.module = module;
            pg_desc.miss.entryFunctionName = name[i].c_str();

            char log[2048];
            size_t log_size = sizeof(log);
            OPTIX_CHECK(optixProgramGroupCreate(
                context,
                &pg_desc,
                1,
                &pg_options,
                log,
                &log_size,
                &module_pg.missPGs[i]
            ));
            if (log_size > 1)
                cout << "Optix miss program log: " << log << endl;
        }
    }

    // create hitgroup program
    {
        string name[2] = { "__closesthit__radiance", "__closesthit__shadow" };
        for (int i = 0; i < 2; i++)
        {
            OptixProgramGroupOptions pg_options = {};
            OptixProgramGroupDesc pg_desc = {};

            pg_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
            pg_desc.hitgroup.moduleCH = module;
            pg_desc.hitgroup.entryFunctionNameCH = name[i].c_str();
            pg_desc.hitgroup.moduleAH = nullptr;
            pg_desc.hitgroup.entryFunctionNameAH = nullptr;

            char log[2048];
            size_t log_size = sizeof(log);
            OPTIX_CHECK(optixProgramGroupCreate(
                context,
                &pg_desc,
                1,
                &pg_options,
                log,
                &log_size,
                &module_pg.hitgroupPGs[i]
            ));
            if (log_size > 1)
                cout << "Optix hitgroup program log: " << log << endl;
        }
    }

    module_pgs.push_back(module_pg);
}

void OptixRayTracer::create_pipeline(const vector<string>& ptxs)
{
    pipeline_compile_options = {};
    pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipeline_compile_options.usesMotionBlur = false;
    pipeline_compile_options.numPayloadValues = 2;
    pipeline_compile_options.numAttributeValues = 2;
    pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    pipeline_compile_options.pipelineLaunchParamsVariableName = "params";

    OptixPipelineLinkOptions pipeline_link_options = {};
    pipeline_link_options.maxTraceDepth = 2;

    pipelines.resize(ptxs.size());
    for (size_t i = 0; i < ptxs.size();i++)
    {
        create_module(ptxs[i]);

        char log[2048];
        size_t log_size = sizeof(log);
        OPTIX_CHECK(optixPipelineCreate(
            context,
            &pipeline_compile_options,
            &pipeline_link_options,
            (OptixProgramGroup*)&module_pgs[i],
            5,
            log,
            &log_size,
            &pipelines[i]
        ));
        if (log_size > 1)
            cout << "Optix pipeline log: " << log << endl;

        OPTIX_CHECK(optixPipelineSetStackSize(pipelines[i], 2048, 2048, 2048, 1));
    }
}

OptixTraversableHandle OptixRayTracer::build_as_from_input(const vector<OptixBuildInput>& inputs, GPUMemory<unsigned char>& as_buffer, bool compact, bool update)
{
    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
    if (compact) accel_options.buildFlags |= OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    else accel_options.buildFlags |= OPTIX_BUILD_FLAG_ALLOW_UPDATE;
    accel_options.motionOptions.numKeys = 1;  // disable motion
    accel_options.operation = update ? OPTIX_BUILD_OPERATION_UPDATE : OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes blas_buffer_sizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(
        context,
        &accel_options,
        inputs.data(),
        (unsigned)inputs.size(),
        &blas_buffer_sizes
    ));

    size_t temp_size = update ? blas_buffer_sizes.tempUpdateSizeInBytes : blas_buffer_sizes.tempSizeInBytes;
    GPUMemory<unsigned char> temp_buffer(temp_size);

    OptixTraversableHandle traversable{ 0 };
    if (compact)
    {
        GPUMemory<uint64_t> compacted_size_buffer(1);
        OptixAccelEmitDesc emit_desc;
        emit_desc.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
        emit_desc.result = (CUdeviceptr)compacted_size_buffer.data();

        GPUMemory<unsigned char> output_buffer(blas_buffer_sizes.outputSizeInBytes);
        OPTIX_CHECK(optixAccelBuild(
            context,
            stream,
            &accel_options,
            inputs.data(),
            (unsigned)inputs.size(),
            (CUdeviceptr)temp_buffer.data(),
            blas_buffer_sizes.tempSizeInBytes,
            (CUdeviceptr)output_buffer.data(),
            blas_buffer_sizes.outputSizeInBytes,
            &traversable,
            &emit_desc,
            1
        ));
        checkCudaErrors(cudaDeviceSynchronize());

        uint64_t compacted_size;
        compacted_size_buffer.copy_to_host(&compacted_size);

        as_buffer.resize(compacted_size);
        OPTIX_CHECK(optixAccelCompact(
            context,
            stream,
            traversable,
            (CUdeviceptr)as_buffer.data(),
            compacted_size,
            &traversable
        ));
        checkCudaErrors(cudaDeviceSynchronize());
    }
    else
    {
        as_buffer.resize(blas_buffer_sizes.outputSizeInBytes);
        OPTIX_CHECK(optixAccelBuild(
            context,
            stream,
            &accel_options,
            inputs.data(),
            (unsigned)inputs.size(),
            (CUdeviceptr)temp_buffer.data(),
            blas_buffer_sizes.tempSizeInBytes,
            (CUdeviceptr)as_buffer.data(),
            blas_buffer_sizes.outputSizeInBytes,
            &traversable,
            nullptr,
            0
        ));
        checkCudaErrors(cudaDeviceSynchronize());
    }
    return traversable;
}

void OptixRayTracer::build_gas()
{
    int mesh_num = (int)scene->meshes.size();
    GScene& gscene = scene->gscene;

    d_vertices.resize(mesh_num);
    gas_input_flags.resize(mesh_num);
    gas_inputs.resize(mesh_num);
    gas_traversable.resize(mesh_num);
    gas_buffer.resize(mesh_num);

    for (int i = 0; i < mesh_num; i++)
    {
        gas_inputs[i] = {};
        gas_inputs[i].type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

        d_vertices[i] = (CUdeviceptr)gscene.vertex_buffer[i].data();
        // d_indices[i] = (CUdeviceptr)gscene.index_buffer[i].data();

        gas_inputs[i].triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
        gas_inputs[i].triangleArray.vertexStrideInBytes = sizeof(float3);
        gas_inputs[i].triangleArray.numVertices = (unsigned)scene->meshes[i]->vertices.size();
        gas_inputs[i].triangleArray.vertexBuffers = &d_vertices[i];

        gas_inputs[i].triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
        gas_inputs[i].triangleArray.indexStrideInBytes = sizeof(int3);
        gas_inputs[i].triangleArray.numIndexTriplets = (unsigned)scene->meshes[i]->indices.size();
        // gas_inputs[i].triangleArray.indexBuffer = d_indices[i];
        gas_inputs[i].triangleArray.indexBuffer = (CUdeviceptr)gscene.index_buffer[i].data();

        gas_input_flags[i] = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;
        gas_inputs[i].triangleArray.flags = &gas_input_flags[i];
        gas_inputs[i].triangleArray.numSbtRecords = 1;
        gas_inputs[i].triangleArray.sbtIndexOffsetBuffer = 0;
        gas_inputs[i].triangleArray.sbtIndexOffsetSizeInBytes = 0;
        gas_inputs[i].triangleArray.sbtIndexOffsetStrideInBytes = 0;

        bool compact = !scene->has_animation();
        gas_traversable[i] = build_as_from_input({ gas_inputs[i] }, gas_buffer[i], compact, false);
    }
}

void OptixRayTracer::build_ias()
{
    int instance_num = (int)scene->instances.size();

    ias_instances.resize(instance_num);
    for (int i = 0; i < instance_num; i++)
    {
        ias_instances[i].instanceId = i;
        ias_instances[i].sbtOffset = i * RAY_TYPE_COUNT;
        ias_instances[i].visibilityMask = 255;
        ias_instances[i].flags = OPTIX_INSTANCE_FLAG_NONE;
        ias_instances[i].traversableHandle = gas_traversable[scene->instances[i]];
        memcpy(ias_instances[i].transform, &(scene->instance_transforms[i]), sizeof(float) * 12);
    }
    ias_instances_buffer.resize_and_copy_from_host(ias_instances);

    ias_input.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
    ias_input.instanceArray.numInstances = instance_num;
    ias_input.instanceArray.instances = (CUdeviceptr)ias_instances_buffer.data();

    bool compact = !scene->has_animation();
    ias_traversable = build_as_from_input({ ias_input }, ias_buffer, compact, false);
}

void OptixRayTracer::build_sbt()
{
    sbts.resize(module_pgs.size());
    raygen_sbt.resize(module_pgs.size());
    miss_sbt.resize(module_pgs.size());
    hitgroup_sbt.resize(module_pgs.size());

    GScene& gscene = scene->gscene;

    for (int i = 0; i < module_pgs.size(); i++)
    {
        RaygenSBTRecord raygen_record;
        OPTIX_CHECK(optixSbtRecordPackHeader(module_pgs[i].raygenPG, &raygen_record));
        raygen_sbt[i].resize_and_copy_from_host(&raygen_record, 1);
        sbts[i].raygenRecord = (CUdeviceptr)raygen_sbt[i].data();

        vector<MissSBTRecord> miss_records;
        for (int j = 0; j < RAY_TYPE_COUNT; j++)
        {
            MissSBTRecord miss_record;
            OPTIX_CHECK(optixSbtRecordPackHeader(module_pgs[i].missPGs[j], &miss_record));
            miss_records.push_back(miss_record);
        }
        miss_sbt[i].resize_and_copy_from_host(miss_records);
        sbts[i].missRecordBase = (CUdeviceptr)miss_sbt[i].data();
        sbts[i].missRecordStrideInBytes = sizeof(MissSBTRecord);
        sbts[i].missRecordCount = (unsigned)miss_records.size();

        int instance_num = (int)scene->instances.size();
        vector<HitgroupSBTRecord> hitgroup_records;
        for (int k = 0; k < instance_num; k++)
        {
            for (int j = 0; j < RAY_TYPE_COUNT; j++)
            {
                HitgroupSBTRecord hitgroup_record;
                OPTIX_CHECK(optixSbtRecordPackHeader(module_pgs[i].hitgroupPGs[j], &hitgroup_record));

                hitgroup_record.data.instance = gscene.instance_buffer.data() + k;
                hitgroup_record.data.light_id = gscene.instance_light_id[k];

                hitgroup_records.push_back(hitgroup_record);
            }
        }
        hitgroup_sbt[i].resize_and_copy_from_host(hitgroup_records);
        sbts[i].hitgroupRecordBase = (CUdeviceptr)hitgroup_sbt[i].data();
        sbts[i].hitgroupRecordStrideInBytes = sizeof(HitgroupSBTRecord);
        sbts[i].hitgroupRecordCount = (unsigned)hitgroup_records.size();
    }
}

OptixRayTracer::OptixRayTracer(const vector<string>& _ptxfiles, shared_ptr<Scene> _scene)
    : scene(_scene)
{
    cout << "Initializing optix..." << endl;
    init_optix();

    cout << "Creating optix context..." << endl;
    create_context();

    cout << "Creating optix pipeline..." << endl;
    std::filesystem::path ptx_folder(PTX_FOLDER);
    vector<string> ptxs;
    for (const auto& ptxfile : _ptxfiles)
    {
        std::filesystem::path ptx_path = ptx_folder / ptxfile;
        string shader = get_ptx_from_file(ptx_path.string());
        ptxs.push_back(shader);
    }
    create_pipeline(ptxs);

    cout << "Building geometry acceleration structure..." << endl;
    build_gas();

    cout << "Building instance acceleration structure..." << endl;
    build_ias();

    cout << "Building shader binding table..." << endl;
    build_sbt();

    cout << "Optix fully set up!" << endl << endl;
}

void OptixRayTracer::update_as()
{
    if (!scene->has_animation()) return;
    if (scene->is_static()) return;

    if (scene->has_bone())
    {
        for (int i = 0; i < (int)scene->meshes.size(); i++)
        {
            if (!scene->meshes[i]->has_bone) continue;
            gas_traversable[i] = build_as_from_input({ gas_inputs[i] }, gas_buffer[i], false, true);
        }
    }
    for (int i = 0; i < (int)scene->instances.size(); i++)
    {
        ias_instances[i].traversableHandle = gas_traversable[scene->instances[i]];
        memcpy(ias_instances[i].transform, &(scene->instance_transforms[i]), sizeof(float) * 12);
    }
    ias_instances_buffer.copy_from_host(ias_instances);
    ias_traversable = build_as_from_input({ ias_input }, ias_buffer, false, true);
}