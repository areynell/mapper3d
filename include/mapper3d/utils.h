#pragma once

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <fstream>
#include <small_gicp/registration/registration_helper.hpp>
#include <sophus/se3.hpp>
#include <string>
#include <vector>

namespace mapper3d {
namespace utils {

std::vector<Eigen::Vector3f> point_cloud_to_vec(const pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud);
pcl::PointCloud<pcl::PointXYZ>::Ptr point_vec_to_cloud(const std::vector<Eigen::Vector3f>& point_vec);

Sophus::SE3d eigen_to_sophus(const Eigen::Matrix4d& eigen_matrix);

Eigen::Isometry3d sophus_to_isometry(const Sophus::SE3d& pose);

double deg2rad(const double deg);
double rad2deg(const double rad);

// Trajectory output in the TUM format that evo consumes:
// timestamp tx ty tz qx qy qz qw
// One pose per line, timestamp in seconds.
std::ofstream open_tum_file(const std::string& path, const std::string& description);

// Write a single pose to a TUM trajectory file.
void write_tum_pose(std::ostream& out, const double timestamp, const Sophus::SE3d& pose);

}  // namespace utils
}  // namespace mapper3d