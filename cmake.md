ch5 stereo
project(STEREOVISION)
cmake_minimum_required(VERSION 3.0)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
set(CMAKE_BUILD_TYPE "Debug")
find_package(Pangolin REQUIRED)
find_package(OpenCV REQUIRES)
include_directories(${OpenCV_INCLUDE_DIRS} "/usr/include/eigen3" ${Pangolin_INCLUDE_DIRS})
add_executable(stereoVision stereoVision.cpp)
target_link_libraries(stereoVision ${OpenCV_LIBS} ${Pangolin_LIBRARIES})

