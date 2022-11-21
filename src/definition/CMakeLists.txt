add_library(definition STATIC "definition.cu")
target_link_libraries(definition PUBLIC ${ZLIB_LIB} ${TINY_CUDA_NN_LIB})