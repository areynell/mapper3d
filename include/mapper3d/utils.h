#pragma once

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <fstream>
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

// The 6x6 covariance of a gtsam::Pose3 factor with independent axes, from standard deviations (rotation in rad,
// translation in m). GTSAM orders a pose's six components rotation first, then translation.
Eigen::Matrix<double, 6, 6> gtsam_pose_covariance(const Eigen::Vector3d& rotation_sigmas,
                                                  const Eigen::Vector3d& translation_sigmas);

// The share of the source cloud's points that registration matched. 0 when the cloud is empty.
double inlier_ratio(const size_t num_inliers, const size_t num_points);

// The registration error averaged over the matched points. Infinity when nothing matched.
double error_per_inlier(const double error, const size_t num_inliers);

// Trajectory output in the TUM format that evo consumes:
// timestamp tx ty tz qx qy qz qw
// One pose per line, timestamp in seconds.
std::ofstream open_tum_file(const std::string& path, const std::string& description);

// Write a single pose to a TUM trajectory file.
void write_tum_pose(std::ostream& out, const double timestamp, const Sophus::SE3d& pose);

}  // namespace utils
}  // namespace mapper3d