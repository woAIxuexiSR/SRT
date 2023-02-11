set(SHADERS 
    "pathTracer/pathTracer.cu"
    "pathTracer/pathTracer_NEE.cu"
    "bdpt/bdpt.cu"
    )
set(SHADERS_HEADERS "")

message("hello" "${MODULE_TARGET_DIR}")
NVCUDA_COMPILE_MODULE(
  SOURCES ${SHADERS}
  DEPENDENCIES ${SHADERS_HEADERS}
  TARGET_PATH "${PTX_OUTPUT_PATH}"
  EXTENSION ".ptx"
  GENERATED_FILES PROGRAM_MODULES
  NVCC_OPTIONS "--ptx" "--machine=64" "--gpu-architecture=compute_86" "--use_fast_math" "--relocatable-device-code=true" "--generate-line-info" "-Wno-deprecated-gpu-targets" "--expt-relaxed-constexpr" "-I${OptiX_INCLUDE}" "-I${CMAKE_SOURCE_DIR}/src/device_include"
)

add_executable(func func.cu ${PROGRAM_MODULES})

# add_library(func_lib STATIC bdpt.cu)