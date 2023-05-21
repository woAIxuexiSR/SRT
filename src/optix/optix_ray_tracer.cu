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

    int num_devices;
    checkCudaErrors(cudaGetDeviceCount(&num_devices));
    if (num_devices == 0)
    {
        cout << "ERROR::No CUDA capable devices found!" << endl;
        return;
    }
    cout << "Found " << num_devices << " CUDA devices" << endl;

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
        return;
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

    // create module
    char log[2048];
    size_t log_size = sizeof(log);
    OPTIX_CHECK(optixModuleCreateFromPTX(
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

void OptixRayTracer::build_as()
{
    int mesh_num = (int)scene->meshes.size();
    DeviceSceneData& d_scene = scene->d_scene;

    vector<OptixBuildInput> triangle_input(mesh_num);
    vector<CUdeviceptr> d_vertices(mesh_num);
    vector<CUdeviceptr> d_indices(mesh_num);
    vector<uint32_t> triangle_input_flags(mesh_num);

    for (int i = 0; i < mesh_num; i++)
    {
        triangle_input[i] = {};
        triangle_input[i].type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

        d_vertices[i] = (CUdeviceptr)d_scene.vertex_buffer[i].data();
        d_indices[i] = (CUdeviceptr)d_scene.index_buffer[i].data();

        triangle_input[i].triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
        triangle_input[i].triangleArray.vertexStrideInBytes = sizeof(float3);
        triangle_input[i].triangleArray.numVertices = (unsigned)scene->meshes[i]->vertices.size();
        triangle_input[i].triangleArray.vertexBuffers = &d_vertices[i];

        triangle_input[i].triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
        triangle_input[i].triangleArray.indexStrideInBytes = sizeof(int3);
        triangle_input[i].triangleArray.numIndexTriplets = (unsigned)scene->meshes[i]->indices.size();
        triangle_input[i].triangleArray.indexBuffer = d_indices[i];

        triangle_input_flags[i] = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;
        triangle_input[i].triangleArray.flags = &triangle_input_flags[i];
        triangle_input[i].triangleArray.numSbtRecords = 1;
        triangle_input[i].triangleArray.sbtIndexOffsetBuffer = 0;
        triangle_input[i].triangleArray.sbtIndexOffsetSizeInBytes = 0;
        triangle_input[i].triangleArray.sbtIndexOffsetStrideInBytes = 0;
    }

    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    accel_options.motionOptions.numKeys = 1;  // disable motion
    accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes blas_buffer_sizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(
        context,
        &accel_options,
        triangle_input.data(),
        (unsigned)mesh_num,
        &blas_buffer_sizes
    ));

    GPUMemory<uint64_t> compacted_size_buffer(1);
    OptixAccelEmitDesc emit_desc;
    emit_desc.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emit_desc.result = (CUdeviceptr)compacted_size_buffer.data();

    GPUMemory<unsigned char> temp_buffer(blas_buffer_sizes.tempSizeInBytes);
    GPUMemory<unsigned char> output_buffer(blas_buffer_sizes.outputSizeInBytes);

    OPTIX_CHECK(optixAccelBuild(
        context,
        stream,
        &accel_options,
        triangle_input.data(),
        (unsigned)mesh_num,
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

void OptixRayTracer::build_sbt()
{
    sbts.resize(module_pgs.size());
    raygen_sbt.resize(module_pgs.size());
    miss_sbt.resize(module_pgs.size());
    hitgroup_sbt.resize(module_pgs.size());

    DeviceSceneData& d_scene = scene->d_scene;

    for (int i = 0; i < module_pgs.size(); i++)
    {
        RaygenSBTRecord raygen_record;
        OPTIX_CHECK(optixSbtRecordPackHeader(module_pgs[i].raygenPG, &raygen_record));
        raygen_sbt[i].resize_and_copy_from_host(&raygen_record, 1);
        sbts[i].raygenRecord = (CUdeviceptr)raygen_sbt[i].data();

        vector<MissSBTRecord> miss_records;
        for (int j = 0; j < 2; j++)
        {
            MissSBTRecord miss_record;
            OPTIX_CHECK(optixSbtRecordPackHeader(module_pgs[i].missPGs[j], &miss_record));
            miss_records.push_back(miss_record);
        }
        miss_sbt[i].resize_and_copy_from_host(miss_records);
        sbts[i].missRecordBase = (CUdeviceptr)miss_sbt[i].data();
        sbts[i].missRecordStrideInBytes = sizeof(MissSBTRecord);
        sbts[i].missRecordCount = (unsigned)miss_records.size();

        int mesh_num = (int)scene->meshes.size();
        vector<HitgroupSBTRecord> hitgroup_records;
        for (int k = 0; k < mesh_num; k++)
        {
            for (int j = 0; j < 2; j++)
            {
                HitgroupSBTRecord hitgroup_record;
                OPTIX_CHECK(optixSbtRecordPackHeader(module_pgs[i].hitgroupPGs[j], &hitgroup_record));

                hitgroup_record.data.vertex = (float3*)d_scene.vertex_buffer[k].data();
                hitgroup_record.data.index = (uint3*)d_scene.index_buffer[k].data();
                hitgroup_record.data.normal = (float3*)d_scene.normal_buffer[k].data();
                hitgroup_record.data.texcoord = (float2*)d_scene.texcoord_buffer[k].data();
                hitgroup_record.data.mesh_id = k;
                hitgroup_record.data.light_id = d_scene.meshid_to_lightid[k];
                hitgroup_record.data.mat = d_scene.material_buffer.data() + scene->meshes[k]->material_id;

                if (scene->meshes[k]->texture_id >= 0)
                {
                    hitgroup_record.data.has_texture = true;
                    hitgroup_record.data.texture = d_scene.texture_objects[scene->meshes[k]->texture_id];
                }
                else
                    hitgroup_record.data.has_texture = false;

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
    std::filesystem::path ptx_folder("ptx");
    vector<string> ptxs;
    for (const auto& ptxfile : _ptxfiles)
    {
        std::filesystem::path ptx_path = ptx_folder / ptxfile;
        string shader = get_ptx_from_file(ptx_path.string());
        ptxs.push_back(shader);
    }
    create_pipeline(ptxs);

    cout << "Building acceleration structure..." << endl;
    build_as();

    cout << "Building shader binding table..." << endl;
    build_sbt();

    cout << "Optix fully set up!" << endl;
}