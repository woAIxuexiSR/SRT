add_executable(srtracer main.cu)
target_link_libraries(srtracer PUBLIC renderer)

add_executable(test test.cu)
target_link_libraries(test PUBLIC renderer)