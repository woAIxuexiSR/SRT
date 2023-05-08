add_library(optix STATIC "optix_ray_tracer.cu")
target_link_libraries(optix PUBLIC definition scene ${CUDA_LIB})