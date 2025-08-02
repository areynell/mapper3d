#pragma once

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <Eigen/Core>
#include <small_gicp/registration/registration_helper.hpp>
#include <sophus/se3.hpp>
#include <vector>

namespace mapper3d {
namespace utils {

std::vector<Eigen::Vector3f> point_cloud_to_vec(const pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud);
pcl::PointCloud<pcl::PointXYZ>::Ptr point_vec_to_cloud(const std::vector<Eigen::Vector3f>& point_vec);

Sophus::SE3d eigen_to_sophus(const Eigen::Matrix4d& eigen_matrix);

Sophus::SE3d constrain_pose_to_se2(const Sophus::SE3d& pose);

double deg2rad(const double deg);
double rad2deg(const double rad);

}  // namespace utils
}  // namespace mapper3d