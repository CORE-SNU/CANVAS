cmake_minimum_required(VERSION 2.8.3)
project(gazebo_rossim)
add_compile_options(-std=c++17)

find_package(catkin REQUIRED COMPONENTS
  gazebo_ros
  gazebo_plugins
  geometry_msgs
  nav_msgs
)

find_package(gazebo REQUIRED)

catkin_package(
  INCLUDE_DIRS 
    include
  CATKIN_DEPENDS
    gazebo_ros
    gazebo_plugins
    geometry_msgs
    nav_msgs
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${GAZEBO_INCLUDE_DIRS}
)

link_directories(${GAZEBO_LIBRARY_DIRS})
list(APPEND CMAKE_CXX_FLAGS "${GAZEBO_CXX_FLAGS}")

