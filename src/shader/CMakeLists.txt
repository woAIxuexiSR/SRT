file(GLOB INTEGRATOR_SOURCES "integrator/*.cu")
file(GLOB NPR_SOURCES "npr/*.cu")
file(GLOB WAVEFRONT_SOURCES "wavefront/*.cu")
set(SHADERS ${INTEGRATOR_SOURCES} ${NPR_SOURCES} ${WAVEFRONT_SOURCES})

file(GLOB MATH_HEADERS "../device_include/math/*.h")
file(GLOB PARAMS_HEADERS "../device_include/params/*.h")
file(GLOB SCENE_HEADERS "../device_include/scene/*.h")
file(GLOB HELPER_HEADERS "../device_include/*.h")
set(SHADER_DEPENDENCIES ${MATH_HEADERS} ${PARAMS_HEADERS} ${SCENE_HEADERS} ${HELPER_HEADERS})

NVCUDA_COMPILE_MODULE(
  SOURCES ${SHADERS}
  DEPENDENCIES ${SHADER_DEPENDENCIES}
  TARGET_PATH "${PTX_OUTPUT_PATH}"
  EXTENSION ".ptx"
  GENERATED_FILES PROGRAM_MODULES
  NVCC_OPTIONS "--ptx" "--machine=64" "--gpu-architecture=compute_${CMAKE_CUDA_ARCHITECTURES}" "--use_fast_math" "--relocatable-device-code=true" "--generate-line-info" "-Wno-deprecated-gpu-targets" "--expt-relaxed-constexpr" "-I${OptiX_INCLUDE}" "-I${CMAKE_SOURCE_DIR}/src/device_include"
)

add_executable(func func.cu ${PROGRAM_MODULES})