add_library(renderer STATIC "renderer.cu")

if(MSVC)
	target_link_libraries(renderer PUBLIC gui importer passes -WHOLEARCHIVE:$<TARGET_FILE:gui> -WHOLEARCHIVE:$<TARGET_FILE:process>)
else()
	target_link_libraries(renderer PUBLIC "-Wl,--whole-archive" gui importer passes "-Wl,--no-whole-archive")
endif()
