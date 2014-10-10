cmake_minimum_required(VERSION 2.8)

project (Maroilles)

option(BUILD_TESTS "build unit tests" ON)
option(BUILD_EXAMPLES "build example applications" OFF)

# Handle platform specifics
if ( "${CMAKE_SYSTEM_NAME}" MATCHES "Windows" )
	SET( MAROILLES_CURRENT_SYSTEM "WINDOWS" )
    ADD_DEFINITIONS( -D__WINDOWS__ )
elseif( "${CMAKE_SYSTEM_NAME}" MATCHES "Linux" )
	SET( MAROILLES_CURRENT_SYSTEM "LINUX" )
    ADD_DEFINITIONS( -D__LINUX__ )
elseif( "${CMAKE_SYSTEM_NAME}" MATCHES "Darwin" )
	SET( MAROILLES_CURRENT_SYSTEM "MACOSX" )
    ADD_DEFINITIONS( -D__MACOSX__ )
endif()

add_subdirectory(lib)
#add_subdirectory(apps)

if( BUILD_TESTS )
    add_subdirectory( tests )
endif( BUILD_TESTS )

if( BUILD_EXAMPLES )
    add_subdirectory(examples)
endif( BUILD_EXAMPLES )
