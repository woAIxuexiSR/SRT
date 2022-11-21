add_library(renderer STATIC "renderer.cu")
target_link_libraries(renderer PUBLIC definition scene ${TINY_CUDA_NN_LIB} ${CUDA_LIB} optix gui)