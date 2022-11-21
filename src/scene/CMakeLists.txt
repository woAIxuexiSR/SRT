add_library(scene STATIC scene.cu film.cu)
target_link_libraries(scene PUBLIC definition)