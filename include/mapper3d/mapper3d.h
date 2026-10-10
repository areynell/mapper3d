#pragma once

#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/registration/gicp.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <fstream>
#include <kiss_matcher/KISSMatcher.hpp>
#include <memory>
#include <optional>
#include <small_gicp/ann/gaussian_voxelmap.hpp>
#include <small_gicp/pcl/pcl_point.hpp>
#include <small_gicp/pcl/pcl_point_traits.hpp>
#include <small_gicp/pcl/pcl_proxy.hpp>
#include <small_gicp/registration/registration_result.hpp>
#include <sophus/se3.hpp>
#include <string>
#include <vector>

#include "mapper3d/graph_optimizer.h"

namespace mapper3d {

struct LocalRegistrationParams {
    int num_threads = 8;                       // Number of threads for GICP
    int num_neighbors_for_covariance = 20;     // Number of neighbors for covariance estimation
    size_t wide_search_scans_after_reset = 5;  // Number of scans to allow a wide search window after local map reset

    // At 10 Hz the normal gap is 0.1 s and one dropped scan gives 0.2 s.
    double dropped_scan_interval = 0.15;  // (s)

    // Size of the rolling local map, in insertions. A voxel no scan has touched for lru_horizon
    // insertions is deleted, and the delete pass runs every lru_clear_cycle.
    size_t lru_horizon = 100;
    size_t lru_clear_cycle = 10;

    double min_inlier_ratio = 0.70;      // result.num_inliers / points in the scan (healthy: 0.85 to 1.0)
    double max_error_per_inlier = 0.60;  // result.error / result.num_inliers (healthy: 0.06 to 0.30)

    // Failures in a row before the global registration runs. Coasting never updates the local map, so it
    // cannot recover on its own.
    size_t max_consecutive_failures = 3;

    // A global registration answer faster than this is rejected. A speed, not a distance, so the limit grows
    // with the gap between scans.
    double max_plausible_sensor_speed = 50.0;  // (m/s), about 180 km/h
};

struct Mapper3DParams {
    // IMPORTANT: This should be the same as the voxel size used in Mapper3D and LoopClosureDetector
    double voxel_size = 0.5;  // Voxel resolution for downsampling (m)

    double z_height_threshold = -1.0;  // Z height threshold for filtering (m)
    double min_sensor_distance = 3.0;  // Distance threshold for filtering points near the sensor (m)

    // Keyframe selection: Only a frame that has moved far enough from the previous keyframe becomes a node.
    double keyframe_distance_threshold = 1.0;                   // Min translation between keyframes (m)
    double keyframe_rotation_threshold = utils::deg2rad(10.0);  // Min rotation between keyframes (rad)

    // Per-scan trajectories written to disk in the TUM format, for offline evaluation with evo.
    std::string lidar_odometry_trajectory_file = "lidar_odometry_tum.txt";
    std::string online_trajectory_file = "online_estimate_tum.txt";

    // Standard deviations of one keyframe-to-keyframe odometry edge.
    double x_odom_uncertainty = 0.05;                      // X position uncertainty for lidar odometry (m)
    double y_odom_uncertainty = 0.05;                      // Y position uncertainty for lidar odometry (m)
    double z_odom_uncertainty = 0.05;                      // Z position uncertainty for lidar odometry (m)
    double roll_odom_uncertainty = utils::deg2rad(0.25);   // Roll orientation uncertainty for lidar odometry (rad)
    double pitch_odom_uncertainty = utils::deg2rad(0.25);  // Pitch orientation uncertainty for lidar odometry (rad)
    double yaw_odom_uncertainty = utils::deg2rad(0.25);    // Yaw orientation uncertainty for lidar odometry (rad)

    LocalRegistrationParams local_reg_params;
    GraphOptimizerParams graph_optimizer_params;
};

class Mapper3D {
   public:
    Mapper3D() = delete;
    Mapper3D(const Mapper3DParams &params);

    void spin_once(const pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud, const double point_cloud_timestamp);

    Sophus::SE3d sensor_pose_odom_frame() const;
    Sophus::SE3d sensor_pose_map_frame() const;

    const pcl::PointCloud<pcl::PointXYZ>::Ptr downsampled_point_cloud() const;

    gtsam::Values get_unoptimized_poses();

    gtsam::Values get_optimized_poses();

    gtsam::NonlinearFactorGraph get_graph();

    pcl::PointCloud<pcl::PointXYZ>::ConstPtr get_map();

    void get_graph_visualization_data(VisualizationData &data);

   private:
    void check_homogeneous_point_format(const pcl::PointCloud<pcl::PointXYZ>::Ptr &point_cloud) const;

    bool should_add_keyframe(const Sophus::SE3d &sensor_pose) const;

    Sophus::SE3d estimate_lidar_odometry(const Sophus::SE3d &prev_pose, const Sophus::SE3d::Tangent &prev_twist,
                                         const pcl::PointCloud<pcl::PointXYZ>::Ptr &point_cloud, const double dt);

    Sophus::SE3d::Tangent estimate_twist(const Sophus::SE3d &prev_pose, const Sophus::SE3d &pose, const double dt);

    Sophus::SE3d predict_sensor_pose(const Sophus::SE3d &pose, const Sophus::SE3d::Tangent &twist,
                                     const double dt) const;

    void estimate_point_cloud_covariances(small_gicp::PointCloudProxy<pcl::PointXYZ> &source);

    small_gicp::RegistrationResult align_to_local_map(const small_gicp::PointCloudProxy<pcl::PointXYZ> &source,
                                                      const Sophus::SE3d &initial_guess, const double dt) const;

    bool use_wide_correspondence_search(const double dt) const;

    small_gicp::GaussianVoxelMap::Ptr make_local_map() const;

    void reset_local_map(const Sophus::SE3d &pose, const small_gicp::PointCloudProxy<pcl::PointXYZ> &source);

    std::optional<Sophus::SE3d> global_registration_pose(const Sophus::SE3d &prev_pose,
                                                         const pcl::PointCloud<pcl::PointXYZ>::Ptr &point_cloud,
                                                         const double dt);

    std::optional<Sophus::SE3d> recover_with_global_registration(
        const Sophus::SE3d &prev_pose, const pcl::PointCloud<pcl::PointXYZ>::Ptr &point_cloud,
        const small_gicp::PointCloudProxy<pcl::PointXYZ> &source, const double dt);

    pcl::PointCloud<pcl::PointXYZ>::Ptr filter_point_cloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud);

    pcl::PointCloud<pcl::PointXYZ>::Ptr remove_points_below_z(const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
                                                              const double z_threshold);

    pcl::PointCloud<pcl::PointXYZ>::Ptr remove_points_near_sensor(const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
                                                                  const double min_distance);

    bool registration_is_acceptable(const small_gicp::RegistrationResult &result, const size_t num_points) const;

    double z_height_threshold_;
    double min_sensor_distance_;
    pcl::VoxelGrid<pcl::PointXYZ> voxel_grid_;

    small_gicp::GaussianVoxelMap::Ptr local_map_;

    double local_map_voxel_size_;
    size_t lru_horizon_;
    size_t lru_clear_cycle_;

    kiss_matcher::KISSMatcherConfig global_registration_config_;
    kiss_matcher::KISSMatcher global_registration_;

    std::vector<Eigen::Matrix4d> point_cloud_covariances_;
    int registration_num_threads_;
    int num_neighbors_for_covariance_;
    size_t wide_search_scans_after_reset_;
    size_t scans_since_map_reset_;
    double dropped_scan_interval_;

    double min_inlier_ratio_;
    double max_error_per_inlier_;
    size_t consecutive_registration_failures_;
    size_t max_consecutive_registration_failures_;
    double max_plausible_sensor_speed_;

    size_t registration_failure_count_;
    size_t coasted_scan_count_;
    size_t relocalization_count_;
    size_t map_reset_count_;

    pcl::PointCloud<pcl::PointXYZ>::Ptr prev_downsampled_point_cloud_;
    pcl::PointCloud<pcl::PointXYZ>::Ptr downsampled_point_cloud_;

    Sophus::SE3d prev_sensor_pose_odom_frame_;
    Sophus::SE3d sensor_pose_odom_frame_;
    Sophus::SE3d::Tangent prev_twist_;
    double prev_point_cloud_timestamp_;

    Sophus::SE3d sensor_pose_map_frame_;

    double keyframe_distance_threshold_;
    double keyframe_rotation_threshold_;
    Sophus::SE3d latest_keyframe_pose_odom_frame_;
    // Set when a point cloud fails registration or arrives after dropped scans, and reset when a keyframe is added.
    // So it says whether anything went wrong between the previous keyframe and the next one.
    bool keyframe_edge_degraded_;

    Eigen::Matrix<double, 6, 6> odometry_covariance_;

    GraphOptimizer graph_optimizer_;

    std::ofstream lidar_odometry_trajectory_file_;
    std::ofstream online_trajectory_file_;

    size_t point_cloud_id_;
    size_t keyframe_id_;
};

}  // namespace mapper3d