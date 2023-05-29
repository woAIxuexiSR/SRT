file(GLOB INTEGRATOR_SOURCES "integrator/*.cu")
file(GLOB NPR_SOURCES "npr/*.cu")
file(GLOB WAVEFRONT_SOURCES "wavefront/*.cu")

set(SHADERS ${INTEGRATOR_SOURCES} ${NPR_SOURCES} ${WAVEFRONT_SOURCES})
set(SHADERS_HEADERS "")

NVCUDA_COMPILE_MODULE(
  SOURCES ${SHADERS}
  DEPENDENCIES ${SHADERS_HEADERS}
  TARGET_PATH "${PTX_OUTPUT_PATH}"
  EXTENSION ".ptx"
  GENERATED_FILES PROGRAM_MODULES
  NVCC_OPTIONS "--ptx" "--machine=64" "--gpu-architecture=compute_86" "--use_fast_math" "--relocatable-device-code=true" "--generate-line-info" "-Wno-deprecated-gpu-targets" "--expt-relaxed-constexpr" "-I${OptiX_INCLUDE}" "-I${CMAKE_SOURCE_DIR}/src/device_include"
)

add_executable(func func.cu ${PROGRAM_MODULES})