cmake_minimum_required(VERSION 2.8.3)
#include($ENV{ROS_ROOT}/core/rosbuild/rosbuild.cmake)
project(loam_velodyne)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11 -march=native -DEIGEN_DONT_PARALLELIZE")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O3 -Wall")

find_package(catkin REQUIRED COMPONENTS
  roscpp
  sensor_msgs
  tf
  geometry_msgs
  std_msgs
  nav_msgs
  quadrotor_msgs
  laser_geometry
  message_generation
  genmsg
)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

#find_package(OpenCV 2)
find_package(OpenCV REQUIRED)
find_package(PCL 1.7 REQUIRED)
find_package(Eigen3  REQUIRED)
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${PCL_INCLUDE_DIRS}
)

include_directories(
	include
	${catkin_INCLUDE_DIRS} 
	${PCL_INCLUDE_DIRS})

#include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})

set(CMAKE_MODULE_PATH /usr/local/lib/cmake/)

catkin_package(
    )

add_executable(velo_scanRegistration src/scanRegistration.cpp)
add_executable(velo_laserOdometry src/laserOdometry.cpp)
add_executable(velo_laserMapping src/laserMapping.cpp)
add_executable(velo_transformMaintenance src/transformMaintenance.cpp)
#add_executable(laser_vins_compare src/laser_vins_compare.cpp)
#add_executable(globalCorrection src/globalCorrection.cpp)

target_link_libraries(velo_laserMapping ${catkin_LIBRARIES} ${OpenCV_LIBS} ${PCL_LIBRARIES})
target_link_libraries(velo_laserOdometry ${catkin_LIBRARIES} ${OpenCV_LIBS} ${PCL_LIBRARIES})
target_link_libraries(velo_scanRegistration ${catkin_LIBRARIES} ${OpenCV_LIBS} ${PCL_LIBRARIES})
target_link_libraries(velo_transformMaintenance ${catkin_LIBRARIES} ${OpenCV_LIBS} ${PCL_LIBRARIES})
#target_link_libraries(globalCorrection ${catkin_LIBRARIES} ${OpenCV_LIBS} ${PCL_LIBRARIES})
#target_link_libraries(laser_vins_compare ${catkin_LIBRARIES} ${OpenCV_LIBS} ${PCL_LIBRARIES})
