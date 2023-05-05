#include "optix_ray_tracer.h"
#include <optix_function_table_definition.h>

string getPtxFromFile(const string& filename)
{
    std::ifstream fs(filename, std::ios::in);
    if (!fs.is_open()) {
        cout << "Could not open file " << filename << endl;
        exit(-1);
    }

    std::stringstream ss;
    ss << fs.rdbuf();
    string str = ss.str();
    fs.close();

    return str;
}

/* the only allowed shader function names */
const string raygenName = "__raygen__";
const string missName[2] = { "__miss__radiance", "__miss__shadow" };
const std::pair<string, string> hitgroupName[2] = {
    {"__closesthit__radiance", "__anyhit__radiance"},
    {"__closesthit__shadow", "__anyhit__shadow"}
};

void OptixRayTracer::initOptix()
{
    checkCudaErrors(cudaFree(0));

    int numDevices;
    checkCudaErrors(cudaGetDeviceCount(&numDevices));
    if (numDevices == 0)
    {
        cout << "no CUDA capable devices found!" << endl;
        exit(-1);
    }
    cout << "found " << numDevices << " CUDA devices" << endl;

    OPTIX_CHECK(optixInit());
    cout << "successfully initialized optix ..." << endl;
}

void OptixRayTracer::createContext()
{
    const int deviceID = 0;
    checkCudaErrors(cudaSetDevice(deviceID));
    checkCudaErrors(cudaStreamCreate(&stream));

    cudaDeviceProp deviceProps;
    checkCudaErrors(cudaGetDeviceProperties(&deviceProps, deviceID));
    cout << "running on device " << deviceProps.name << endl;

    CUcontext cudaContext;
    CUresult cuRes = cuCtxGetCurrent(&cudaContext);
    if (cuRes != CUDA_SUCCESS)
    {
        cout << "could not get CUDA context" << endl;
        exit(-1);
    }

    OPTIX_CHECK(optixDeviceContextCreate(cudaContext, 0, &optixContext));
}

void OptixRayTracer::createModule(const string& ptx, OptixPipelineCompileOptions& pipelineCompileOptions, vector<OptixProgramGroup>& programGroups)
{
    // create module
    OptixModule module;

    OptixModuleCompileOptions moduleCompileOptions = {};
    moduleCompileOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
    // moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;
    // moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
    // moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;

    {
        char log[2048];
        size_t logSize = sizeof(log);
        OPTIX_CHECK(optixModuleCreateFromPTX(optixContext, &moduleCompileOptions, &pipelineCompileOptions, ptx.c_str(), ptx.size(), log, &logSize, &module));
        if (logSize > 1)
            cout << "Optix log : " << log << endl;
    }

    ModuleProgramGroup modulePG;

    // create raygen programs
    {
        OptixProgramGroupOptions pgOptions = {};
        OptixProgramGroupDesc pgDesc = {};

        pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        pgDesc.raygen.module = module;
        pgDesc.raygen.entryFunctionName = raygenName.c_str();

        char log[2048];
        size_t logSize = sizeof(log);
        OPTIX_CHECK(optixProgramGroupCreate(optixContext, &pgDesc, 1, &pgOptions, log, &logSize, &modulePG.raygenPG));
        if (logSize > 1)
            cout << "Optix log : " << log << endl;

        programGroups.push_back(modulePG.raygenPG);
    }

    // create miss programs
    {
        for (int i = 0; i < 2; i++)
        {
            OptixProgramGroupOptions pgOptions = {};
            OptixProgramGroupDesc pgDesc = {};

            pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
            pgDesc.miss.module = module;
            pgDesc.miss.entryFunctionName = missName[i].c_str();

            char log[2048];
            size_t logSize = sizeof(log);
            OPTIX_CHECK(optixProgramGroupCreate(optixContext, &pgDesc, 1, &pgOptions, log, &logSize, &modulePG.missPGs[i]));
            if (logSize > 1)
                cout << "Optix log : " << log << endl;

            programGroups.push_back(modulePG.missPGs[i]);
        }
    }

    // create hitgroup programs
    {
        for (int i = 0; i < 2; i++)
        {
            OptixProgramGroupOptions pgOptions = {};
            OptixProgramGroupDesc pgDesc = {};

            pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
            pgDesc.hitgroup.moduleCH = module;
            pgDesc.hitgroup.entryFunctionNameCH = hitgroupName[i].first.c_str();
            pgDesc.hitgroup.moduleAH = module;
            pgDesc.hitgroup.entryFunctionNameAH = hitgroupName[i].second.c_str();

            char log[2048];
            size_t logSize = sizeof(log);
            OPTIX_CHECK(optixProgramGroupCreate(optixContext, &pgDesc, 1, &pgOptions, log, &logSize, &modulePG.hitgroupPGs[i]));
            if (logSize > 1)
                cout << "Optix log : " << log << endl;

            programGroups.push_back(modulePG.hitgroupPGs[i]);
        }
    }

    modulePGs.push_back(modulePG);
}

void OptixRayTracer::createPipelines()
{
    OptixPipelineCompileOptions pipelineCompileOptions = {};
    pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipelineCompileOptions.usesMotionBlur = false;
    pipelineCompileOptions.numPayloadValues = 2;
    pipelineCompileOptions.numAttributeValues = 2;
    pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    pipelineCompileOptions.pipelineLaunchParamsVariableName = "params";

    OptixPipelineLinkOptions pipelineLinkOptions = {};
    pipelineLinkOptions.maxTraceDepth = 2;

    pipelines.resize(modulePTXs.size());
    for (size_t i = 0; i < modulePTXs.size(); i++)
    {
        vector<OptixProgramGroup> programGroups;
        createModule(modulePTXs[i], pipelineCompileOptions, programGroups);

        char log[2048];
        size_t logSize = sizeof(log);
        OPTIX_CHECK(optixPipelineCreate(optixContext, &pipelineCompileOptions, &pipelineLinkOptions, programGroups.data(), (unsigned)programGroups.size(), log, &logSize, &pipelines[i]));
        if (logSize > 1)
            cout << "Optix log : " << log << endl;

        OPTIX_CHECK(optixPipelineSetStackSize(pipelines[i], 2048, 2048, 2048, 1));
    }
}

void OptixRayTracer::buildAccel()
{
    int meshNum = (int)scene->meshes.size();

    vertexBuffer.resize(meshNum);
    indexBuffer.resize(meshNum);
    texcoordBuffer.resize(meshNum);
    normalBuffer.resize(meshNum);

    vector<OptixBuildInput> triangleInput(meshNum);
    vector<CUdeviceptr> d_vertices(meshNum);
    vector<CUdeviceptr> d_indices(meshNum);
    vector<uint32_t> triangleInputFlags(meshNum);

    for (int i = 0; i < meshNum; i++)
    {
        TriangleMesh& mesh = *(scene->meshes[i]);
        vertexBuffer[i].resize_and_copy_from_host(mesh.vertices);
        indexBuffer[i].resize_and_copy_from_host(mesh.indices);
        if (!mesh.texcoords.empty())
            texcoordBuffer[i].resize_and_copy_from_host(mesh.texcoords);
        if (!mesh.normals.empty())
            normalBuffer[i].resize_and_copy_from_host(mesh.normals);

        triangleInput[i] = {};
        triangleInput[i].type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

        d_vertices[i] = (CUdeviceptr)vertexBuffer[i].data();
        d_indices[i] = (CUdeviceptr)indexBuffer[i].data();

        triangleInput[i].triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
        triangleInput[i].triangleArray.vertexStrideInBytes = sizeof(float3);
        triangleInput[i].triangleArray.numVertices = (unsigned)mesh.vertices.size();
        triangleInput[i].triangleArray.vertexBuffers = &d_vertices[i];

        triangleInput[i].triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
        triangleInput[i].triangleArray.indexStrideInBytes = sizeof(uint3);
        triangleInput[i].triangleArray.numIndexTriplets = (unsigned)mesh.indices.size();
        triangleInput[i].triangleArray.indexBuffer = d_indices[i];

        // triangleInputFlags[i] = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;
        triangleInputFlags[i] = 0;
        triangleInput[i].triangleArray.flags = &triangleInputFlags[i];
        triangleInput[i].triangleArray.numSbtRecords = 1;
        triangleInput[i].triangleArray.sbtIndexOffsetBuffer = 0;
        triangleInput[i].triangleArray.sbtIndexOffsetSizeInBytes = 0;
        triangleInput[i].triangleArray.sbtIndexOffsetStrideInBytes = 0;
    }

    OptixAccelBuildOptions accelOptions = {};
    accelOptions.buildFlags = OPTIX_BUILD_FLAG_NONE | OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    accelOptions.motionOptions.numKeys = 1;
    accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes blasBufferSizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(optixContext, &accelOptions, triangleInput.data(), meshNum, &blasBufferSizes));

    GPUMemory<uint64_t> compactedSizeBuffer(1);
    OptixAccelEmitDesc emitDesc;
    emitDesc.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emitDesc.result = (CUdeviceptr)compactedSizeBuffer.data();

    GPUMemory<unsigned char> tempBuffer(blasBufferSizes.tempSizeInBytes);
    GPUMemory<unsigned char> outputBuffer(blasBufferSizes.outputSizeInBytes);

    OPTIX_CHECK(optixAccelBuild(
        optixContext,
        stream,
        &accelOptions,
        triangleInput.data(),
        meshNum,
        (CUdeviceptr)tempBuffer.data(),
        blasBufferSizes.tempSizeInBytes,
        (CUdeviceptr)outputBuffer.data(),
        blasBufferSizes.outputSizeInBytes,
        &traversable,
        &emitDesc,
        1));
    checkCudaErrors(cudaDeviceSynchronize());

    uint64_t compactedSize;
    compactedSizeBuffer.copy_to_host(&compactedSize);

    asBuffer.resize(compactedSize);
    OPTIX_CHECK(optixAccelCompact(optixContext, stream, traversable, (CUdeviceptr)asBuffer.data(), compactedSize, &traversable));
    checkCudaErrors(cudaDeviceSynchronize());
}

void OptixRayTracer::createTextures()
{
    int numTextures = (int)scene->textures.size();

    textureArrays.resize(numTextures);
    textureObjects.resize(numTextures);

    for (int i = 0; i < numTextures; i++)
    {
        Texture* texture = scene->textures[i].get();

        cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<uchar4>();

        int width = texture->resolution.x;
        int height = texture->resolution.y;
        int nComponents = 4;
        int pitch = width * nComponents * sizeof(unsigned char);

        cudaArray_t& pixelArray = textureArrays[i];
        checkCudaErrors(cudaMallocArray(&pixelArray, &channel_desc, width, height));
        checkCudaErrors(cudaMemcpy2DToArray(pixelArray, 0, 0, texture->pixels, pitch, pitch, height, cudaMemcpyHostToDevice));

        cudaResourceDesc res_desc = {};
        res_desc.resType = cudaResourceTypeArray;
        res_desc.res.array.array = pixelArray;

        cudaTextureDesc tex_desc = {};
        tex_desc.addressMode[0] = cudaAddressModeWrap;
        tex_desc.addressMode[1] = cudaAddressModeWrap;
        tex_desc.filterMode = cudaFilterModeLinear;
        tex_desc.readMode = cudaReadModeNormalizedFloat;
        tex_desc.normalizedCoords = 1;
        tex_desc.maxAnisotropy = 1;
        tex_desc.maxMipmapLevelClamp = 99;
        tex_desc.minMipmapLevelClamp = 0;
        tex_desc.mipmapFilterMode = cudaFilterModePoint;
        tex_desc.borderColor[0] = 1.0f;
        tex_desc.sRGB = 0;

        cudaTextureObject_t cuda_tex = 0;
        checkCudaErrors(cudaCreateTextureObject(&cuda_tex, &res_desc, &tex_desc, nullptr));
        textureObjects[i] = cuda_tex;
    }
}

void OptixRayTracer::buildSBT()
{
    sbts.resize(modulePGs.size());
    raygenRecordsBuffer.resize(modulePGs.size());
    missRecordsBuffer.resize(modulePGs.size());
    hitgroupRecordsBuffer.resize(modulePGs.size());

    for (int i = 0; i < modulePGs.size(); i++)
    {
        RaygenSBTRecord raygenRec;
        OPTIX_CHECK(optixSbtRecordPackHeader(modulePGs[i].raygenPG, &raygenRec));
        raygenRecordsBuffer[i].resize_and_copy_from_host(&raygenRec, 1);
        sbts[i].raygenRecord = (CUdeviceptr)raygenRecordsBuffer[i].data();

        vector<MissSBTRecord> missRecs;
        for (int j = 0; j < 2; j++)
        {
            MissSBTRecord rec;
            OPTIX_CHECK(optixSbtRecordPackHeader(modulePGs[i].missPGs[j], &rec));
            missRecs.push_back(rec);
        }
        missRecordsBuffer[i].resize_and_copy_from_host(missRecs);
        sbts[i].missRecordBase = (CUdeviceptr)missRecordsBuffer[i].data();
        sbts[i].missRecordStrideInBytes = sizeof(MissSBTRecord);
        sbts[i].missRecordCount = (unsigned)missRecs.size();

        int meshNum = (int)scene->meshes.size();
        vector<HitgroupSBTRecord> hitgroupRecs;
        for (int k = 0; k < meshNum; k++)
        {
            for (int j = 0; j < 2; j++)
            {
                HitgroupSBTRecord rec;
                OPTIX_CHECK(optixSbtRecordPackHeader(modulePGs[i].hitgroupPGs[j], &rec));

                rec.data.vertex = (float3*)vertexBuffer[k].data();
                rec.data.index = (uint3*)indexBuffer[k].data();
                rec.data.normal = (float3*)normalBuffer[k].data();
                rec.data.texcoord = (float2*)texcoordBuffer[k].data();
                rec.data.mesh_id = k;
                rec.data.mat = *scene->materials[scene->meshes[k]->material_id];
                if (scene->meshes[k]->texture_id >= 0)
                {
                    rec.data.has_texture = true;
                    rec.data.texture = textureObjects[scene->meshes[k]->texture_id];
                }
                else
                    rec.data.has_texture = false;

                hitgroupRecs.push_back(rec);
            }
        }
        hitgroupRecordsBuffer[i].resize_and_copy_from_host(hitgroupRecs);
        sbts[i].hitgroupRecordBase = (CUdeviceptr)hitgroupRecordsBuffer[i].data();
        sbts[i].hitgroupRecordStrideInBytes = sizeof(HitgroupSBTRecord);
        sbts[i].hitgroupRecordCount = (unsigned)hitgroupRecs.size();
    }
}

void OptixRayTracer::generateLight()
{
    vector<float3> lightVertices;
    vector<float3> lightNormals;
    vector<uint3> lightIndices;
    vector<float3> lightEmissions;
    for (int i = 0; i < scene->meshes.size(); i++)
    {
        TriangleMesh* mesh = scene->meshes[i].get();
        if (!scene->materials[mesh->material_id]->is_emissive())
            continue;
        int vertexOffset = (int)lightVertices.size();
        lightVertices.insert(lightVertices.end(), mesh->vertices.begin(), mesh->vertices.end());
        if (mesh->normals.size() > 0)
            lightNormals.insert(lightNormals.end(), mesh->normals.begin(), mesh->normals.end());
        else
        {
            for (int j = 0; j < mesh->vertices.size(); j++)
                lightNormals.push_back(make_float3(0.0f));
        }

        for (int j = 0; j < mesh->indices.size(); j++)
        {
            uint3 index = mesh->indices[j];
            lightIndices.push_back(index + vertexOffset);
            lightEmissions.push_back(scene->materials[mesh->material_id]->get_emission());
        }
    }
    int numTriangles = (int)lightIndices.size();
    vector<float> lightAccumArea(numTriangles);
    float totalArea = 0.0f;
    for (int i = 0; i < numTriangles; i++)
    {
        float3 v0 = lightVertices[lightIndices[i].x];
        float3 v1 = lightVertices[lightIndices[i].y];
        float3 v2 = lightVertices[lightIndices[i].z];
        float3 e0 = v1 - v0;
        float3 e1 = v2 - v0;
        float3 normal = cross(e0, e1);
        totalArea += length(normal) * 0.5f;
        lightAccumArea[i] = totalArea;
    }

    lightVertexBuffer.resize_and_copy_from_host(lightVertices);
    lightNormalBuffer.resize_and_copy_from_host(lightNormals);
    lightIndexBuffer.resize_and_copy_from_host(lightIndices);
    lightAccumAreaBuffer.resize_and_copy_from_host(lightAccumArea);
    lightEmissionBuffer.resize_and_copy_from_host(lightEmissions);
    light.set(numTriangles, lightVertexBuffer.data(), lightNormalBuffer.data(), lightIndexBuffer.data(),
        lightAccumAreaBuffer.data(), lightEmissionBuffer.data(), totalArea);
}

OptixRayTracer::OptixRayTracer(const vector<string>& _ptxfiles, const Scene* _scene): scene(_scene)
{
    std::filesystem::path ptxFolder("ptx");
    for (const auto& ptxfile : _ptxfiles)
    {
        std::filesystem::path ptxPath = ptxFolder / ptxfile;
        string shader = getPtxFromFile(ptxPath.string());
        modulePTXs.push_back(shader);
    }

    cout << "initializing optix ..." << endl;
    initOptix();

    cout << "creating optix context ..." << endl;
    createContext();

    cout << "setting up optix pipline ..." << endl;
    createPipelines();

    cout << "building acceleration structure ..." << endl;
    buildAccel();

    cout << "creating textures ..." << endl;
    createTextures();

    cout << "building SBT ..." << endl;
    buildSBT();

    cout << "generating light ..." << endl;
    generateLight();

    cout << "Optix 7 Renderer fully set up!" << endl;
}