add_library(gui STATIC "gui.cpp" ${GLAD_SRC})
target_link_libraries(gui PUBLIC ${GLFW_LIB} ${OPENGL_LIB} definition)