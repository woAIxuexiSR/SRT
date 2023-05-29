add_library(renderer STATIC "renderer.cu" "pbrtparse.cu")

if(MSVC)
	target_link_libraries(renderer PUBLIC gui process -WHOLEARCHIVE:$<TARGET_FILE:gui> -WHOLEARCHIVE:$<TARGET_FILE:process>)
else()
	target_link_libraries(renderer PUBLIC "-Wl,--whole-archive" gui process "-Wl,--no-whole-archive")
endif()
