include_directories( ${LIBMAROILLES_SOURCE_DIR}/include )

add_executable( example_0 main.cpp )
target_link_libraries( example_0 maroilles )
