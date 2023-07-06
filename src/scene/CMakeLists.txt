add_library(scene STATIC "animation.cpp" "mesh.cpp" "texture.cpp" "scene.cu")
target_link_libraries(scene PUBLIC definition)