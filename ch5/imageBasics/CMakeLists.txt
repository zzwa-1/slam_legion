#project(imageBasics)

#add_executable(imageBasics imageBasics.cpp)
# 链接OpenCV库
#target_link_libraries(imageBasics ${OpenCV_LIBS})

#add_executable(undistortImage undistortImage.cpp)
# 链接OpenCV库
#target_link_libraries(undistortImage ${OpenCV_LIBS})
set(CMAKE_CXX_FLAGS "-std=c++11")
cmake_minimum_required(VERSION 3.0)
project(IMAGEBASICS)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
set(CMAKE_BUILD_TYPE "Debug")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(imageBasics imageBasics.cpp)
target_link_libraries(imageBasics ${OpenCV_LIBS})
