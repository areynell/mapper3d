cmake_minimum_required(VERSION 3.22)
project(mapper3d)

# set(CMAKE_CXX_COMPILER clang++)
set(CMAKE_CXX_STANDARD 17)

# # AddressSanitizer flags
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fno-omit-frame-pointer -g")
# set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=address")
# set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fsanitize=address")

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -mtune=native")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
# set(CMAKE_CXX_FLAGS "-O0 -g -ggdb -fsanitize=address")
# set(CMAKE_CXX_FLAGS "-O0 -g -ggdb")

# # Multiple build types for different use cases
# if(CMAKE_BUILD_TYPE STREQUAL "Debug")
#     # set(CMAKE_CXX_FLAGS_DEBUG "-g3 -O0 -ggdb -DDEBUG_MODE")
#     set(CMAKE_CXX_FLAGS_DEBUG "-g -O2 -ggdb -DDEBUG_MODE")

# elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
#     set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")  # Portable

# elseif(CMAKE_BUILD_TYPE STREQUAL "ReleaseNative")
#     set(CMAKE_CXX_FLAGS_RELEASENATIVE "-O3 -DNDEBUG -march=native -mtune=native")  # Maximum performance

# elseif(CMAKE_BUILD_TYPE STREQUAL "Profile")
#     set(CMAKE_CXX_FLAGS_PROFILE "-O2 -g -DNDEBUG -march=native")  # Fast + debuggable
# endif()

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
find_package(visualization_msgs REQUIRED)

find_package(Eigen3 REQUIRED)
find_package(Sophus CONFIG REQUIRED)
find_package(TBB REQUIRED)
find_package(teaserpp REQUIRED)
find_package(robin REQUIRED)
find_package(kiss_matcher REQUIRED)
find_package(PCL CONFIG REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(OpenCV REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(GTSAM CONFIG REQUIRED)
find_package(small_gicp REQUIRED)
find_package(blosc CONFIG REQUIRED)
find_package(OpenVDB CONFIG REQUIRED)

 # Include directories
include_directories(
  include
  3rdparty/small_gicp/include
  3rdparty/Scancontext
  ${PCL_INCLUDE_DIRS}
  ${Sophus_INCLUDE_DIRS}
)

# Create libraries
add_library(utils src/utils.cpp)
target_link_libraries(utils
  ${PCL_LIBRARIES}
  Eigen3::Eigen
  small_gicp::small_gicp
)

add_library(common_dependencies INTERFACE)
target_link_libraries(common_dependencies INTERFACE
  ${PCL_LIBRARIES}
  ${OpenCV_LIBS}
  Eigen3::Eigen
  Sophus::Sophus
  utils
)
target_include_directories(common_dependencies INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

add_library(loop_closure_detector
  src/loop_closure_detector.cpp
  3rdparty/Scancontext/Scancontext.cpp
)
target_link_libraries(loop_closure_detector
  ${PCL_LIBRARIES}
  ${pcl_conversions_LIBRARIES}
  ${OpenCV_LIBS}
  cv_bridge::cv_bridge
  Sophus::Sophus
  Eigen3::Eigen
  TBB::tbb
  teaserpp::teaser_registration
  teaserpp::teaser_io
  robin::robin
  kiss_matcher::kiss_matcher_core
  common_dependencies
)
target_include_directories(loop_closure_detector PUBLIC
  ${PCL_INCLUDE_DIRS}
  ${pcl_conversions_INCLUDE_DIRS}
)

add_library(graph_optimizer src/graph_optimizer.cpp)
target_link_libraries(graph_optimizer
  loop_closure_detector
  Sophus::Sophus
  Eigen3::Eigen
  gtsam
  OpenVDB::openvdb
  blosc_static # Required for OpenVDB
  common_dependencies
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
  common_dependencies
)
target_include_directories(mapper3d PUBLIC
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
  visualization_msgs
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

# Install launch files
install(
  DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()