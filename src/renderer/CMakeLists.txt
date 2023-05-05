add_library(renderer STATIC "renderer.cu")
target_link_libraries(renderer PUBLIC process)