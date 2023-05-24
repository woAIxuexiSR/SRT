cmake_minimum_required(VERSION 3.20)

project(SRT LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 86)
# set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_BUILD_TYPE Debug)

set(EXTENSIONS_DIR "${CMAKE_SOURCE_DIR}/extensions")

# opengl
find_package(OpenGL REQUIRED)
set(OPENGL_LIB "OpenGL::GL")

# glad
set(GLAD_INCLUDE "${EXTENSIONS_DIR}/glad")
set(GLAD_SRC "${EXTENSIONS_DIR}/glad/glad.c")

# glfw
add_subdirectory("${EXTENSIONS_DIR}/glfw" EXCLUDE_FROM_ALL)
set(GLFW_INCLUDE "${EXTENSIONS_DIR}/glfw/include")
set(GLFW_LIB "glfw")

# imgui
set(IMGUI_INCLUDE "${EXTENSIONS_DIR}/imgui/include")
file(GLOB IMGUI_SRC "${EXTENSIONS_DIR}/imgui/src/*.cpp")
# implot
set(IMPLOT_INCLUDE "${EXTENSIONS_DIR}/implot/include")
file(GLOB IMPLOT_SRC "${EXTENSIONS_DIR}/implot/src/*.cpp")
# imguizmo
set(IMZMO_INCLUDE "${EXTENSIONS_DIR}/imguizmo")
set(IMZMO_SRC "${EXTENSIONS_DIR}/imguizmo/ImGuizmo.cpp")

# cuda
find_package(CUDA REQUIRED)
find_package(CUDAToolkit 10.0 REQUIRED)
find_program(BIN2C bin2c)
set(CUDA_INCLUDE "${CUDA_TOOLKIT_ROOT_DIR}/include")
set(CUDA_LIB "${CUDA_LIBRARIES}" "${CUDA_CUDA_LIBRARY}" "cuda" "cublas")

# assimp
add_subdirectory("${EXTENSIONS_DIR}/assimp" EXCLUDE_FROM_ALL)
set(ASSIMP_INCLUDE "${EXTENSIONS_DIR}/assimp/include")
set(ASSIMP_LIB "assimp")

# stb_image
set(STB_INCLUDE "${EXTENSIONS_DIR}/stb_image")
# zlib
find_package(ZLIB REQUIRED)
set(ZLIB_LIB "${ZLIB_LIBRARIES}")
# tiny exr
set(TINY_EXR_INCLUDE "${EXTENSIONS_DIR}/tinyexr")
set(TINY_PLY_INCLUDE "${EXTENSIONS_DIR}/tinyply")
# json
set(JSON_INCLUDE "${EXTENSIONS_DIR}/json")

# tiny cuda nn
add_subdirectory(${EXTENSIONS_DIR}/tiny-cuda-nn EXCLUDE_FROM_ALL)
set(TINY_CUDA_NN_INCLUDE "${EXTENSIONS_DIR}/tiny-cuda-nn/include")
set(TINY_CUDA_NN_LIB "tiny-cuda-nn")

# optix
set(OptiX_INSTALL_DIR $ENV{OptiX_INSTALL_DIR} CACHE PATH "Path to OptiX installed location.")
list(APPEND CMAKE_MODULE_PATH "${OptiX_INSTALL_DIR}/SDK/CMake")
find_package(OptiX REQUIRED)


FUNCTION(NVCUDA_COMPILE_MODULE)
  if (NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
    message(FATAL_ERROR "ERROR: Only 64-bit programs supported.")
  endif()

  set(options "")
  set(oneValueArgs TARGET_PATH GENERATED_FILES EXTENSION)
  set(multiValueArgs NVCC_OPTIONS SOURCES DEPENDENCIES)

  CMAKE_PARSE_ARGUMENTS(NVCUDA_COMPILE_MODULE "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
  
  if (NOT WIN32) # Do not create a folder with the name ${ConfigurationName} under Windows.
    # Under Linux make sure the target directory exists. 
    FILE(MAKE_DIRECTORY ${NVCUDA_COMPILE_MODULE_TARGET_PATH})
  endif()
  
  # Custom build rule to generate either *.ptx or *.optixir files from *.cu files.
  FOREACH(input ${NVCUDA_COMPILE_MODULE_SOURCES})
    get_filename_component(input_we "${input}" NAME_WE)
    get_filename_component(ABS_PATH "${input}" ABSOLUTE)
    string(REPLACE "${CMAKE_CURRENT_SOURCE_DIR}/" "" REL_PATH "${ABS_PATH}")

    # Generate the output *.ptx or *.optixir files directly into the executable's selected target directory.
    set(output "${NVCUDA_COMPILE_MODULE_TARGET_PATH}/${input_we}${NVCUDA_COMPILE_MODULE_EXTENSION}")
    # message("output = ${output}")

    LIST(APPEND OUTPUT_FILES "${output}")
    
    # This prints the standalone NVCC command line for each CUDA file.
    # CUDAToolkit_NVCC_EXECUTABLE has been set with FindCUDAToolkit.cmake in CMake 3.17 and newer.
    message("${CUDAToolkit_NVCC_EXECUTABLE} " "${NVCUDA_COMPILE_MODULE_NVCC_OPTIONS} " "${input} " "-o " "${output}")

    add_custom_command(
      OUTPUT  "${output}"
      DEPENDS "${input}" ${NVCUDA_COMPILE_MODULE_DEPENDENCIES}
      COMMAND ${CUDAToolkit_NVCC_EXECUTABLE} ${NVCUDA_COMPILE_MODULE_NVCC_OPTIONS} "${input}" -o "${output}"
      WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    )
  ENDFOREACH( )

  set(${NVCUDA_COMPILE_MODULE_GENERATED_FILES} ${OUTPUT_FILES} PARENT_SCOPE)
ENDFUNCTION()

set(EXECUTABLE_OUTPUT_PATH "${CMAKE_BINARY_DIR}/bin")
set(LIBRARY_OUTPUT_PATH "${CMAKE_BINARY_DIR}/lib")
set(PTX_OUTPUT_PATH "${CMAKE_BINARY_DIR}/bin/ptx")

add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>)
# add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-G>)

list(APPEND PROJECT_INCLUDE 
    ${GLAD_INCLUDE}
    ${GLFW_INCLUDE}
    ${IMGUI_INCLUDE}
    ${IMPLOT_INCLUDE}
    ${IMZMO_INCLUDE}
    ${CUDA_INCLUDE}
    ${ASSIMP_INCLUDE}
    ${STB_INCLUDE}
    ${TINY_EXR_INCLUDE}
    ${TINY_PLY_INCLUDE}
    ${JSON_INCLUDE}
    ${TINY_CUDA_NN_INCLUDE}
    ${OptiX_INCLUDE}
)
include_directories(${PROJECT_INCLUDE})

add_subdirectory(src)