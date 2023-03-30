add_library(gui STATIC "gui.cpp" ${OPENGL_SRC} ${IMGUI_SRC} ${IMPLOT_SRC})
target_link_libraries(gui PUBLIC ${OPENGL_LIB} definition)