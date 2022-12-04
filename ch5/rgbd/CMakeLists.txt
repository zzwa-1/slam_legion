cmake_minimum_required(VERSION 3.0)

project(JOINMAP)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -Wall ")#设置编译模式输出警告信息
include_directories("/usr/include/eigen3")  #使用绝对路径的方式

set(CMAKE_BUILD_TYPE "Debug")

find_package(Sophus REQUIRED)
include_directories(${Sophus_INCLUDE_DIRS})

find_package(Pangolin REQUIRED)

add_executable(joinMap joinMap.cpp)
target_link_libraries(joinMap ${OpenCV_LIBS} ${Pangolin_LIBRARIES})