add_library(scene STATIC "animation.cpp" "mesh.cpp" "texture.cu" "scene.cu")
target_link_libraries(scene PUBLIC definition)