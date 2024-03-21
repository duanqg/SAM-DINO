cmake_minimum_required(VERSION 3.0.2)
project(SuperVoxel C CXX)

## Compile as C++14
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 14)
SET(CMAKE_BUILD_TYPE "Release")

add_definitions(-DROOT_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/\")


## System dependencies are found with CMake's conventions
find_package(OpenMP QUIET)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++14 -pthread -g -march=native")
set(CMAKE_CXX_FLAGS "${OpenMP_CXX_FLAGS}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}   ${OpenMP_C_FLAGS}")
find_package(PCL REQUIRED)
#find_package(pybind11 REQUIRED)

#add_subdirectory(pybind11)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

include_directories(
        ${PCL_INCLUDE_DIRS}
        codelibrary
)


add_library(supervoxel SHARED mainpcl.cc pca.h)
target_link_libraries(supervoxel ${PCL_LIBRARIES})

add_executable(main main.cc)
target_link_libraries(main ${PCL_LIBRARIES})


add_executable(normal_voxel normal_voxel.cc)
target_link_libraries(normal_voxel ${PCL_LIBRARIES})