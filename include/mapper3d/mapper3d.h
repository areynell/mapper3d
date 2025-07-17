#pragma once

#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/registration/gicp.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <condition_variable>
#include <deque>
#include <kiss_matcher/FasterPFH.hpp>
#include <kiss_matcher/KISSMatcher.hpp>
#include <queue>
#include <small_gicp/pcl/pcl_point.hpp>
#include <small_gicp/pcl/pcl_point_traits.hpp>
#include <small_gicp/pcl/pcl_registration.hpp>
#include <sophus/se3.hpp>
#include <vector>

#include "mapper3d/graph_optimizer.h"

namespace mapper3d {

struct GICPParams {
    double voxel_size = 0.5;                   // Voxel resolution for downsampling (m)
    int num_threads = 8;                       // Number of threads for GICP
    int num_neighbors_for_covariance = 20;     // Number of neighbors for covariance estimation
    double max_correspondence_distance = 1.0;  // Maximum correspondence distance (m)
    std::string registration_type = "VGICP";   // Registration type: "VGICP" or "GICP"
};

struct Mapper3DParams {
    // IMPORTANT: This should be the same as the voxel size used in Mapper3D and LoopClosureDetector
    double voxel_size = 0.5;  // Voxel size for downsampling (m)
    // double normal_radius = 3.0 * voxel_size;
    // double fpfh_radius = 5.0 * voxel_size;
    double z_height_threshold = -1.0;  // Z height threshold for filtering (m)
    double min_sensor_distance = 3.0;  // Distance threshold for filtering points near the sensor (m)

    double x_odom_uncertainty = 0.02;                     // X position uncertainty for lidar odometry (m)
    double y_odom_uncertainty = 0.02;                     // Y position uncertainty for lidar odometry (m)
    double z_odom_uncertainty = 0.02;                     // Z position uncertainty for lidar odometry (m)
    double roll_odom_uncertainty = utils::deg2rad(1.0);   // Roll orientation uncertainty for lidar odometry (rad)
    double pitch_odom_uncertainty = utils::deg2rad(1.0);  // Pitch orientation uncertainty for lidar odometry (rad)
    double yaw_odom_uncertainty = utils::deg2rad(1.0);    // Yaw orientation uncertainty for lidar odometry (rad)

    GICPParams gicp_params;
    GraphOptimizerParams graph_optimizer_params;
};

struct PointCloudFrame {
    pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud;
    Sophus::SE3d sensor_pose;
};

class Mapper3D {
   public:
    Mapper3D() = delete;
    Mapper3D(const Mapper3DParams &params);

    void spin_once(const pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud);

    Sophus::SE3d sensor_pose_odom_frame() const;
    Sophus::SE3d sensor_pose_map_frame() const;

    const pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud() const;

    const std::vector<Eigen::Vector3f> &point_cloud_vec() const;

    const std::vector<Eigen::Vector3f> &kiss_matcher_point_cloud() const;

    const std::vector<Eigen::Vector3f> &kiss_matcher_keypoint_cloud() const;

    gtsam::Values get_unoptimized_poses();

    gtsam::Values get_optimized_poses();

    gtsam::Values get_online_poses();

    gtsam::NonlinearFactorGraph get_graph();

    pcl::PointCloud<pcl::PointXYZ> get_map();

   private:
    Sophus::SE3d estimate_lidar_odometry(const std::vector<Eigen::Vector3f> &prev_point_cloud,
                                         const std::vector<Eigen::Vector3f> &point_cloud,
                                         const Sophus::SE3d &prev_sensor_pose);

    pcl::PointCloud<pcl::PointXYZ>::Ptr filter_point_cloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud);

    pcl::PointCloud<pcl::PointXYZ>::Ptr remove_points_below_z(const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
                                                              const double z_threshold);

    pcl::PointCloud<pcl::PointXYZ>::Ptr remove_points_near_sensor(const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
                                                                  const double min_distance);

    void enqueue_visualization_data(const pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud,
                                    const Sophus::SE3d &sensor_pose);

    // Sophus::SE3d predict_sensor_motion();

    double z_height_threshold_;
    double min_sensor_distance_;
    pcl::VoxelGrid<pcl::PointXYZ> voxel_grid_;

    kiss_matcher::KISSMatcherConfig kiss_config_;
    kiss_matcher::KISSMatcher kiss_matcher_;
    bool constrain_pose_to_se2_;

    pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud_;
    std::vector<Eigen::Vector3f> prev_point_cloud_vec_;
    std::vector<Eigen::Vector3f> point_cloud_vec_;

    Sophus::SE3d prev_sensor_pose_odom_frame_;
    Sophus::SE3d sensor_pose_odom_frame_;
    Sophus::SE3d sensor_pose_map_frame_;

    Eigen::Matrix<double, 6, 6> odometry_covariance_;

    std::vector<Eigen::Vector3f> kiss_matcher_processed_prev_point_cloud_;
    std::vector<Eigen::Vector3f> kiss_matcher_processed_point_cloud_;

    std::vector<Eigen::Vector3f> kiss_matcher_prev_keypoints_;
    std::vector<Eigen::Vector3f> kiss_matcher_keypoints_;

    std::deque<PointCloudFrame> aligned_point_cloud_frames_;
    size_t max_point_cloud_history_;

    GraphOptimizer graph_optimizer_;

    size_t pose_id_;
};

}  // namespace mapper3d