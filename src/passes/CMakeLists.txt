file(GLOB POST_SOURCES "postprocess/*.cu")
file(GLOB NPR_SOURCES "npr/*.cu")
file(GLOB INT_SOURCES "integrator/*.cu")
file(GLOB WAVE_SOURCES "wavefront/*.cu")

add_library(passes STATIC ${POST_SOURCES} ${NPR_SOURCES} ${INT_SOURCES} ${WAVE_SOURCES})
target_link_libraries(passes PUBLIC definition scene optix)