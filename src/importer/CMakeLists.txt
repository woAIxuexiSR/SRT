add_library(importer STATIC "assimp.cu")
target_link_libraries(importer PUBLIC definition scene)