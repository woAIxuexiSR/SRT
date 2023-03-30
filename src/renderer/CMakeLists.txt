add_library(renderer STATIC "renderer.cpp")
target_link_libraries(renderer PUBLIC scene ${TINY_CUDA_NN_LIB} ${CUDA_LIB} optix gui)