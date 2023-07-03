#pragma once

#include <iostream>
#include <cuda_runtime.h>
#include <optix.h>
#include <optix_stubs.h>

#include "my_math.h"
#include "scene/gmaterial.h"
#include "scene/gmesh.h"

inline void help_optix(OptixResult res, const char* func, const char* file, const int line)
{
    if (res != OPTIX_SUCCESS)
    {
        std::cout << "Optix error at " << file << ":" << line << " in " << func << ": " << optixGetErrorName(res) << std::endl;
        exit(-1);
    }
}
#define OPTIX_CHECK(val) help_optix((val), #val, __FILE__, __LINE__)

enum
{
    RADIANCE_RAY_TYPE,
    SHADOW_RAY_TYPE,
    RAY_TYPE_COUNT
};

// module program group
class ModuleProgramGroup
{
public:
    OptixProgramGroup raygenPG;
    OptixProgramGroup missPGs[2];
    OptixProgramGroup hitgroupPGs[2];
};

// sbt record data
template<class T>
struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) SBTRecord
{
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

struct RaygenData {};
struct MissData {};
struct HitgroupData
{
    GInstance* instance;
    int light_id;
};

typedef SBTRecord<RaygenData> RaygenSBTRecord;
typedef SBTRecord<MissData> MissSBTRecord;
typedef SBTRecord<HitgroupData> HitgroupSBTRecord;