set(SHADERS 
    "pathTracer/pathTracer.cu"
    "pathTracer/pathTracer_.cu"
    "pathTracer/lightTracer.cu"
    "pathTracer/directLight.cu"
    "bdpt/lightPath.cu"
    "bdpt/cameraPath.cu"
    )
set(SHADERS_HEADERS "")

if(CMAKE_BUILD_TYPE MATCHES Release)
  message("hello" "${MODULE_TARGET_DIR}")
  NVCUDA_COMPILE_MODULE(
    SOURCES ${SHADERS}
    DEPENDENCIES ${SHADERS_HEADERS}
    TARGET_PATH "${PTX_OUTPUT_PATH}"
    EXTENSION ".ptx"
    GENERATED_FILES PROGRAM_MODULES
    NVCC_OPTIONS "--ptx" "--machine=64" "--gpu-architecture=compute_86" "--use_fast_math" "--relocatable-device-code=true" "--generate-line-info" "-Wno-deprecated-gpu-targets" "--expt-relaxed-constexpr" "-I${OptiX_INCLUDE}" "-I${CMAKE_SOURCE_DIR}/src/device_include"
  )
elseif(CMAKE_BUILD_TYPE MATCHES Debug)
  message("hello" "${MODULE_TARGET_DIR}")
  NVCUDA_COMPILE_MODULE(
  SOURCES ${SHADERS}
  DEPENDENCIES ${SHADERS_HEADERS}
  TARGET_PATH "${PTX_OUTPUT_PATH}"
  EXTENSION ".ptx"
  GENERATED_FILES PROGRAM_MODULES
  NVCC_OPTIONS "--ptx" "--machine=64" "--gpu-architecture=compute_86" "--use_fast_math" "--relocatable-device-code=true" "--generate-line-info" "-Wno-deprecated-gpu-targets" "--expt-relaxed-constexpr" "-I${OptiX_INCLUDE}" "-I${CMAKE_SOURCE_DIR}/src/device_include"
  )
endif()




add_executable(func func.cu ${PROGRAM_MODULES})

# add_library(func_lib STATIC bdpt.cu)