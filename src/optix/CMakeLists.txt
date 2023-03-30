file(GLOB CPP_SOURCES *.cpp)
file(GLOB CU_SOURCES *.cu)
set(INTEGRATER_SOURCES ${CPP_SOURCES} ${CU_SOURCES})

add_library(optix STATIC ${INTEGRATER_SOURCES})
target_link_libraries(optix PUBLIC definition)