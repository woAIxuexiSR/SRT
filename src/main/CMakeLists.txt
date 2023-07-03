add_executable(SuperRay SuperRay.cu)
target_link_libraries(SuperRay PUBLIC renderer)

add_executable(test test.cu)
target_link_libraries(test PUBLIC renderer)