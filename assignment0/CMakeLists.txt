cmake_minimum_required (VERSION 2.8.11)
project (Transformation)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Choose the type of build (Debug or Release)" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release")
endif()

set(EIGEN_DIR "C:/Users/bestr/Documents/Library/Eigen3.4.0")

include_directories("${EIGEN_DIR}/include")

add_executable (Transformation main.cpp)
