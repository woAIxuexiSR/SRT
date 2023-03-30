add_library(scene STATIC scene.cpp film.cu parser.cpp)
target_link_libraries(scene PUBLIC definition)