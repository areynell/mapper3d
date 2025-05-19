#pragma once

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <Eigen/Core>
#include <vector>

namespace mapper3d {
namespace utils {

std::vector<Eigen::Vector3f> point_cloud_to_vec(const pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud);
pcl::PointCloud<pcl::PointXYZ>::Ptr point_vec_to_cloud(const std::vector<Eigen::Vector3f> &point_vec);

double deg2rad(const double deg);
double rad2deg(const double rad);

}  // namespace utils
}  // namespace mapper3d