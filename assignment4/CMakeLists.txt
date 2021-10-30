cmake_minimum_required(VERSION 3.10)
project(BezierCurve)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Choose the type of build (Debug or Release)" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release")
endif()

set(CMAKE_CXX_STANDARD 17)

set(EIGEN_DIR "C:/Users/bestr/Documents/Library/Eigen3.4.0")
set(OPENCV_DIR "C:/Users/bestr/Documents/Library/OpenCV4.5.4")

include_directories("${EIGEN_DIR}/include")
include_directories("${OPENCV_DIR}/include")

add_executable(BezierCurve main.cpp)

if(CMAKE_BUILD_TYPE MATCHES Debug)
    target_link_libraries(BezierCurve
        "${OPENCV_DIR}/win64/vc15/lib/opencv_world454d.lib")
    set(DEBUG_DLL)
    file(GLOB_RECURSE DEBUG_DLL
        ${OPENCV_DIR}/win64/vc15/bin/opencv_world454d.dll
    )
    foreach(file_i ${DEBUG_DLL})
        add_custom_command(TARGET BezierCurve POST_BUILD 
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${file_i}
            ${CMAKE_SOURCE_DIR}/build/$<CONFIGURATION>
        )
    endforeach(file_i)
else()
    target_link_libraries(BezierCurve
        "${OPENCV_DIR}/win64/vc15/lib/opencv_world454.lib")
    set(RELEASE_DLL)
    file(GLOB_RECURSE RELEASE_DLL
        ${OPENCV_DIR}/win64/vc15/bin/opencv_world454.dll
    )
    foreach(file_i ${RELEASE_DLL})
        add_custom_command(TARGET BezierCurve POST_BUILD 
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${file_i}
            ${CMAKE_SOURCE_DIR}/build/$<CONFIGURATION>
        )
    endforeach(file_i)
endif()
