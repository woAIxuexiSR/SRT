# include_directories(helper)
# add_subdirectory(helper)
# list(APPEND SRC_INCLUDES "${CMAKE_CURRENT_SOURCE_DIR}/helper")

# include_directories(math)
# add_subdirectory(math)
# list(APPEND SRC_INCLUDES "${CMAKE_CURRENT_SOURCE_DIR}/math")

# include_directories(material)
# add_subdirectory(material)
# list(APPEND SRC_INCLUDES "${CMAKE_CURRENT_SOURCE_DIR}/material")

# include_directories(scene)
# add_subdirectory(scene)
# list(APPEND SRC_INCLUDES "${CMAKE_CURRENT_SOURCE_DIR}/scene")

# # message("lueluelue")
# include_directories(optixApp)
# add_subdirectory(optixApp)

# include_directories(gui)
# add_subdirectory(gui)

# add_executable(main main.cu)
# target_link_libraries(main PUBLIC defs)

# add_executable(test test.cu)
# target_link_libraries(test PUBLIC defs scene ${TINY_CUDA_NN_LIB} ${CUDA_LIB} optix gui)

include_directories(argparse)
include_directories(device_include)
include_directories(definition)
include_directories(gui)
include_directories(metric)
include_directories(scene)
include_directories(optix)
include_directories(renderer)
add_subdirectory(definition)
add_subdirectory(gui)
add_subdirectory(scene)
add_subdirectory(optix)
add_subdirectory(shader)
add_subdirectory(renderer)

add_executable(hello hello.cu)
target_link_libraries(hello PUBLIC renderer)

add_executable(main main.cu)
target_link_libraries(main PUBLIC renderer)

add_executable(metric metric.cu)
target_link_libraries(metric PUBLIC definition)