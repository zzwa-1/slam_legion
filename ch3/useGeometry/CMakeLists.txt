cmake_minimum_required( VERSION 2.8 )
project( geometry )
set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
# 添加Eigen头文件
include_directories( "/usr/include/eigen3")
add_executable(eigenGeometry eigenGeometry.cpp)
