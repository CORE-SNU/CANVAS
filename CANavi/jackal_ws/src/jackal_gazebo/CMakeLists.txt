cmake_minimum_required(VERSION 2.8.3)
project(jackal_gazebo)

find_package(gazebo REQUIRED)

find_package(catkin REQUIRED COMPONENTS roslaunch
             gazebo_plugins
             gazebo_ros
)
         
catkin_package(
  INCLUDE_DIRS include
#  LIBRARIES restaurant_jackal
#  CATKIN_DEPENDS other_catkin_pkg
  DEPENDS gazebo
)

roslaunch_add_file_check(launch/jackal_world.launch)
roslaunch_add_file_check(launch/scenario.launch)

include_directories(include ${catkin_INCLUDE_DIRS} ${GAZEBO_INCLUDE_DIRS})
link_directories(${GAZEBO_LIBRARY_DIRS})

install(DIRECTORY launch Media models worlds urdf
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)


catkin_install_python(PROGRAMS scripts/odom_pub.py DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

add_library(ActorPlugin_1 SHARED scripts/ActorPlugin.cc)
target_link_libraries(ActorPlugin_1 ${GAZEBO_LIBRARIES})
