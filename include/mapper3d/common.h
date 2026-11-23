#pragma once

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <Eigen/Dense>
#include <sophus/se3.hpp>
#include <utility>
#include <vector>

#include "mapper3d/logging.h"
#include "mapper3d/utils.h"

namespace mapper3d {

struct VisualizationData {
    std::vector<Sophus::SE3d> poses;
    std::vector<std::pair<size_t, size_t>> loop_closure_edges;
    pcl::PointCloud<pcl::PointXYZ>::ConstPtr map;
};

struct Frame {
    size_t keyframe_id;
    size_t point_cloud_id;
    double timestamp;
    Sophus::SE3d pose;
    pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud;
    Eigen::Matrix<double, 6, 6> odometry_covariance;
    // True when a point cloud since the previous keyframe failed registration or arrived after dropped scans. The
    // odometry edge from the previous keyframe to this one is then left open to rejection by GNC.
    bool odometry_degraded;

    Frame()
        : keyframe_id(0),
          point_cloud_id(0),
          timestamp(0.0),
          pose(Sophus::SE3d()),
          point_cloud(new pcl::PointCloud<pcl::PointXYZ>()),
          odometry_covariance(Eigen::Matrix<double, 6, 6>::Zero()),
          odometry_degraded(false) {}

    Frame(size_t keyframe_id, size_t point_cloud_id, double timestamp, const Sophus::SE3d &pose,
          const pcl::PointCloud<pcl::PointXYZ>::Ptr &point_cloud,
          const Eigen::Matrix<double, 6, 6> &odometry_covariance, bool odometry_degraded)
        : keyframe_id(keyframe_id),
          point_cloud_id(point_cloud_id),
          timestamp(timestamp),
          pose(pose),
          point_cloud(point_cloud),
          odometry_covariance(odometry_covariance),
          odometry_degraded(odometry_degraded) {}
};

}  // namespace mapper3d
