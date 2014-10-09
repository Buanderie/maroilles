cmake_minimum_required(VERSION 2.6)

#
project(maroilles_test)

##
include_directories( ${LIBMAROILLES_SOURCE_DIR}/include )

################################
# GTest
################################
find_package(GTest REQUIRED)
#ADD_SUBDIRECTORY (gtest-1.6.0)
#message( ${_SOURCE_DIR} )
enable_testing()
include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})

################################
# Unit Tests
################################

add_test( maroillesTests maroillesTests )
add_test( maroillesTests realtest )

# Add test cpp file
file ( GLOB_RECURSE MAROILLES_TEST_CPP ./*.cpp )
add_executable( maroillesTests  ${MAROILLES_TEST_CPP} )

# Link test executable against gtest & gtest_main
target_link_libraries( maroillesTests gtest gtest_main pthread maroilles )
