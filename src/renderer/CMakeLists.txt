add_library(renderer STATIC "renderer.cu" "pbrtparse.cu")
target_link_libraries(renderer PUBLIC "-Wl,--whole-archive" gui process "-Wl,--no-whole-archive")
