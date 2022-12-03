cmake_minimum_required(VERSION 3.0)
project(TRAJECTORY)
set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
find_package(Sophus REQUIRED)
find_package(Pangolin REQUIRED)
include_directories("/usr/include/eigen3" ${Sophus_INCLUDE_DIRS} ${Pangolin_INCLUDE_DIRS})
add_executable(trajectoryError trajectoryError.cpp)
target_link_libraries(trajectoryError ${Pangolin_LIBRARIES} Sophus::Sophus)

#option(USE_UBUNTU_20 "Set to ON if you are using Ubuntu 20.04" OFF)
#find_package(Pangolin REQUIRED)
#if(USE_UBUNTU_20)
 #   message("You are using Ubuntu 20.04, fmt::fmt will be linked")
  #  find_package(fmt REQUIRED)
   # set(FMT_LIBRARIES fmt::fmt)
#endif()
#include_directories(${Pangolin_INCLUDE_DIRS})
#add_executable(trajectoryError trajectoryError.cpp)
#target_link_libraries(trajectoryError ${Pangolin_LIBRARIES} ${FMT_LIBRARIES})


#cmake_minimum_required(VERSION 3.0)
#project(useSophus)
#set(CMAKE_BUILD_TYPE "Debug")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
# 为使用 sophus，需要使用find_package命令找到它
#find_package(Sophus REQUIRED)

# Eigen
#include_directories("/usr/include/eigen3" ${Sophus_INCLUDE_DIRS})
#add_executable(useSophus useSophus.cpp)
#target_link_libraries(useSophus Sophus::Sophus)

#add_subdirectory(example)