cmake_minimum_required(VERSION 3.22)

# set(CMAKE_CXX_COMPILER clang++)
set(CMAKE_CXX_STANDARD 17)

project(mapper3d)

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(rosbag2_cpp REQUIRED)

find_package(Eigen3 REQUIRED)
find_package(Sophus CONFIG REQUIRED)
find_package(TBB REQUIRED)
find_package(teaserpp REQUIRED)
find_package(robin REQUIRED)
find_package(kiss_matcher REQUIRED)
find_package(PCL CONFIG REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(OpenCV REQUIRED)
find_package(GTSAM CONFIG REQUIRED)
find_package(small_gicp REQUIRED)

# Include directories
include_directories(
  include
  3rdparty/small_gicp/include
  ${PCL_INCLUDE_DIRS}
  ${Sophus_INCLUDE_DIRS}
)

# Create libraries
add_library(utils src/utils.cpp)
target_link_libraries(utils
  ${PCL_LIBRARIES}
  Eigen3::Eigen
)

add_library(loop_closure_detector src/loop_closure_detector.cpp)
target_link_libraries(loop_closure_detector
  ${PCL_LIBRARIES}
  ${OpenCV_LIBS}
  Sophus::Sophus
  Eigen3::Eigen
  TBB::tbb
  teaserpp::teaser_registration
  teaserpp::teaser_io
  robin::robin
  kiss_matcher::kiss_matcher_core
)

add_library(graph_optimizer src/graph_optimizer.cpp)
target_link_libraries(graph_optimizer
  loop_closure_detector
  Sophus::Sophus
  Eigen3::Eigen
  gtsam
)

add_library(mapper3d src/mapper3d.cpp)
target_link_libraries(mapper3d
  utils
  graph_optimizer
  ${PCL_LIBRARIES}
  Sophus::Sophus
  Eigen3::Eigen
  TBB::tbb
  teaserpp::teaser_registration
  teaserpp::teaser_io
  robin::robin
  kiss_matcher::kiss_matcher_core
  small_gicp::small_gicp
)
target_include_directories(mapper3d
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Create executables
add_executable(mapper3d_node src/mapper3d_node.cpp)
target_link_libraries(mapper3d_node
  mapper3d
)
ament_target_dependencies(mapper3d_node
  rclcpp
  tf2
  tf2_ros
  sensor_msgs
  nav_msgs
  geometry_msgs
  pcl_conversions
)

add_executable(kitti_to_ros2_bag src/kitti_to_ros2_bag.cpp)
ament_target_dependencies(kitti_to_ros2_bag
  rclcpp
  sensor_msgs
  rosbag2_cpp
)

# Install targets
install(TARGETS
  mapper3d_node
  mapper3d
  utils
  kitti_to_ros2_bag
  EXPORT export_mapper3d
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()