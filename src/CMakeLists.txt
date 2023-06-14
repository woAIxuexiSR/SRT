include_directories(
    "definition"
    "device_include"
    "gui"
    "optix"
    "passes"
    "renderer"
    "scene"
    "utils"
)

add_subdirectory(definition)
# add_subdirectory(gui)
add_subdirectory(scene)
# add_subdirectory(optix)
# add_subdirectory(passes)
# add_subdirectory(shader)
# add_subdirectory(renderer)

add_subdirectory(main)