cmake_minimum_required(VERSION 2.8)
project (LIBMAROILLES)

# Set custom cmake scripts directory
set( CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};${LIBMAROILLES_SOURCE_DIR}/cmake")

# Set CMake options
#set( USE_CUDA FALSE CACHE BOOL "Use nVidia CUDA as GPGPU API")

option(USE_CLANG "build application with clang" OFF) # OFF is the default

# Configure CLANG
if( USE_CLANG )
    SET (CMAKE_C_COMPILER             "/usr/bin/clang")
    SET (CMAKE_C_FLAGS                "-Wall -std=c99")
    SET (CMAKE_C_FLAGS_DEBUG          "-g")
    SET (CMAKE_C_FLAGS_MINSIZEREL     "-Os -DNDEBUG")
    SET (CMAKE_C_FLAGS_RELEASE        "-O4 -DNDEBUG")
    SET (CMAKE_C_FLAGS_RELWITHDEBINFO "-O2 -g")

    SET (CMAKE_CXX_COMPILER             "/usr/bin/clang++")
    SET (CMAKE_CXX_FLAGS                "-Wall")
    SET (CMAKE_CXX_FLAGS_DEBUG          "-g")
    SET (CMAKE_CXX_FLAGS_MINSIZEREL     "-Os -DNDEBUG")
    SET (CMAKE_CXX_FLAGS_RELEASE        "-O4 -DNDEBUG")
    SET (CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g")

    SET (CMAKE_AR      "/usr/bin/llvm-ar")
    SET (CMAKE_LINKER  "/usr/bin/llvm-ld")
    SET (CMAKE_NM      "/usr/bin/llvm-nm")
    SET (CMAKE_OBJDUMP "/usr/bin/llvm-objdump")
    SET (CMAKE_RANLIB  "/usr/bin/llvm-ranlib")
endif( USE_CLANG )

# Add Dynamic Library Loader dependency (if UNIX-based)
#if( ${MAROILLES_CURRENT_SYSTEM} STREQUAL "LINUX" )
#	set( DEPS "${DEPS};dl" )
#endif()

#Find OpenCL
find_package( OpenCL REQUIRED )
if( OPENCL_FOUND )
include_directories( ${OPENCL_INCLUDE_DIRS} )
set( DEPS "${DEPS};${OPENCL_LIBRARIES}" )
endif( OPENCL_FOUND )

# Set include directories
include_directories( ${LIBMAROILLES_SOURCE_DIR}/include )

file ( GLOB_RECURSE MAROILLES_CPP ./src/*.cpp )
set( ALL_CPP ${MAROILLES_CPP} )
if( ${MAROILLES_CURRENT_SYSTEM} STREQUAL "WINDOWS" )
  add_library( maroilles ${ALL_CPP} )
else()
  add_library( maroilles SHARED ${ALL_CPP} )
endif()

# Link libraries
target_link_libraries( maroilles ${DEPS} )
