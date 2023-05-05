add_library(scene STATIC "mesh.cpp" "texture.cpp" "scene.cpp")
target_link_libraries(scene PUBLIC definition)