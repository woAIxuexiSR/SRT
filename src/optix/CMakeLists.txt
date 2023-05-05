add_library(optix STATIC "optix_ray_tracer.cpp")
target_link_libraries(optix PUBLIC definition scene ${CUDA_LIB})