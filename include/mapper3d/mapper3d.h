#pragma once

#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/registration/gicp.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <deque>
#include <kiss_matcher/FasterPFH.hpp>
#include <kiss_matcher/KISSMatcher.hpp>
#include <small_gicp/pcl/pcl_point.hpp>
#include <small_gicp/pcl/pcl_point_traits.hpp>
#include <small_gicp/pcl/pcl_registration.hpp>
#include <sophus/se3.hpp>
#include <vector>

#include "mapper3d/graph_optimizer.h"

namespace mapper3d {

struct GlobalRegistrationParams {
    // TODO
};

struct LocalRegistrationParams {
    // double voxel_size = 0.5;                   // Voxel resolution for downsampling (m)
    int num_threads = 8;                       // Number of threads for GICP
    int num_neighbors_for_covariance = 20;     // Number of neighbors for covariance estimation
    double max_correspondence_distance = 1.0;  // Maximum correspondence distance (m)
    std::string registration_type = "VGICP";   // Registration type: "VGICP" or "GICP"
    // TODO: Register against the last N frames instead of the last frame
};

struct Mapper3DParams {
    // IMPORTANT: This should be the same as the voxel size used in Mapper3D and LoopClosureDetector
    double voxel_size = 0.5;  // Voxel resolution for downsampling (m)

    double z_height_threshold = -1.0;  // Z height threshold for filtering (m)
    double min_sensor_distance = 3.0;  // Distance threshold for filtering points near the sensor (m)

    // Keyframe selection: Only a frame that has moved far enough from the previous keyframe becomes a node.
    double keyframe_distance_threshold = 1.0;                   // Min translation between keyframes (m)
    double keyframe_rotation_threshold = utils::deg2rad(10.0);  // Min rotation between keyframes (rad)

    double x_odom_uncertainty = 0.02;                     // X position uncertainty for lidar odometry (m)
    double y_odom_uncertainty = 0.02;                     // Y position uncertainty for lidar odometry (m)
    double z_odom_uncertainty = 0.02;                     // Z position uncertainty for lidar odometry (m)
    double roll_odom_uncertainty = utils::deg2rad(1.0);   // Roll orientation uncertainty for lidar odometry (rad)
    double pitch_odom_uncertainty = utils::deg2rad(1.0);  // Pitch orientation uncertainty for lidar odometry (rad)
    double yaw_odom_uncertainty = utils::deg2rad(1.0);    // Yaw orientation uncertainty for lidar odometry (rad)

    LocalRegistrationParams local_reg_params;
    GraphOptimizerParams graph_optimizer_params;
};

class Mapper3D {
   public:
    Mapper3D() = delete;
    Mapper3D(const Mapper3DParams &params);

    void spin_once(const pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud);

    Sophus::SE3d sensor_pose_odom_frame() const;
    Sophus::SE3d sensor_pose_map_frame() const;

    const pcl::PointCloud<pcl::PointXYZ>::Ptr downsampled_point_cloud() const;

    // const std::vector<Eigen::Vector3f> &point_cloud_vec() const;

    const std::vector<Eigen::Vector3f> &global_registration_point_cloud() const;

    const std::vector<Eigen::Vector3f> &global_registration_keypoints() const;

    gtsam::Values get_unoptimized_poses();

    gtsam::Values get_optimized_poses();

    // std::vector<Sophus::SE3d> get_online_poses();

    // std::vector<std::pair<size_t, size_t>> get_loop_closure_edges();

    gtsam::NonlinearFactorGraph get_graph();

    pcl::PointCloud<pcl::PointXYZ>::ConstPtr get_map();

    void get_graph_visualization_data(VisualizationData &data);

   private:
    Sophus::SE3d estimate_lidar_odometry(const pcl::PointCloud<pcl::PointXYZ>::Ptr &prev_point_cloud,
                                         const pcl::PointCloud<pcl::PointXYZ>::Ptr &point_cloud,
                                         const Sophus::SE3d &prev_sensor_pose);

    bool should_add_keyframe(const Sophus::SE3d &sensor_pose) const;

    Sophus::SE3d local_registration(const pcl::PointCloud<pcl::PointXYZ>::Ptr &prev_point_cloud,
                                    const pcl::PointCloud<pcl::PointXYZ>::Ptr &point_cloud,
                                    const Sophus::SE3d &initial_transform_guess);

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

    kiss_matcher::KISSMatcherConfig global_registration_config_;
    kiss_matcher::KISSMatcher global_registration_;
    small_gicp::RegistrationPCL<pcl::PointXYZ, pcl::PointXYZ> local_registration_;

    bool constrain_pose_to_se2_;

    pcl::PointCloud<pcl::PointXYZ>::Ptr prev_downsampled_point_cloud_;
    pcl::PointCloud<pcl::PointXYZ>::Ptr downsampled_point_cloud_;

    Sophus::SE3d prev_sensor_pose_odom_frame_;
    Sophus::SE3d sensor_pose_odom_frame_;
    Sophus::SE3d sensor_pose_map_frame_;

    double keyframe_distance_threshold_;
    double keyframe_rotation_threshold_;
    Sophus::SE3d latest_keyframe_pose_odom_frame_;

    Eigen::Matrix<double, 6, 6> odometry_covariance_;

    std::vector<Eigen::Vector3f> global_registration_processed_prev_point_cloud_;
    std::vector<Eigen::Vector3f> global_registration_processed_point_cloud_;

    std::vector<Eigen::Vector3f> global_registration_prev_keypoints_;
    std::vector<Eigen::Vector3f> global_registration_keypoints_;

    GraphOptimizer graph_optimizer_;

    size_t pose_id_;
};

}  // namespace mapper3d