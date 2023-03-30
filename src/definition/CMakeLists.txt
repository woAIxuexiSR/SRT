add_library(definition STATIC "definition.cpp")
target_link_libraries(definition PUBLIC ${ZLIB_LIB} ${TINY_CUDA_NN_LIB} ${ASSIMP_LIB})