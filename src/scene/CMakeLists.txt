add_library(scene STATIC "mesh.cpp" "texture.cpp" "scene.cu")
target_link_libraries(scene PUBLIC definition)