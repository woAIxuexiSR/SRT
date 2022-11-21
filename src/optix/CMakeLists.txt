file(GLOB INTEGRATER_SOURCES *.cu)

add_library(optix STATIC ${INTEGRATER_SOURCES})
target_link_libraries(optix PUBLIC definition)