add_library(scene STATIC "texture.cpp" "animation.cpp" "assimp.cpp")
target_link_libraries(scene PUBLIC definition)