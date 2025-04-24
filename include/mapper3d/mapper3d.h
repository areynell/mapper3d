#pragma once

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

namespace mapper3d {

struct GICPParams {
    double voxel_size = 0.5;                   // Voxel resolution for downsampling (m)
    int num_threads = 32;                      // Number of threads for GICP
    int num_neighbors_for_covariance = 20;     // Number of neighbors for covariance estimation
    double max_correspondence_distance = 1.0;  // Maximum correspondence distance (m)
    std::string registration_type = "VGICP";   // Registration type: "VGICP" or "GICP"
};

struct Mapper3DParams {
    double voxel_size = 0.5;           // Voxel size for downsampling (m)
    double z_height_threshold = -1.0;  // Z height threshold for filtering (m)

    GICPParams gicp_params;
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

    Sophus::SE3d sensor_pose() const;

    pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud() const;

    const std::vector<Eigen::Vector3f> &kiss_matcher_point_cloud() const;

    const std::vector<Eigen::Vector3f> &kiss_matcher_keypoint_cloud() const;

   private:
    Sophus::SE3d estimate_lidar_odometry(const pcl::PointCloud<pcl::PointXYZ>::Ptr prev_point_cloud,
                                         const pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud,
                                         const Sophus::SE3d &prev_sensor_pose);

    Sophus::SE3d eigen_to_sophus(const Eigen::Matrix4d &eigen_matrix);

    Sophus::SE3d constrain_pose_to_se2(const Sophus::SE3d &pose);

    pcl::PointCloud<pcl::PointXYZ>::Ptr filter_point_cloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud,
                                                           const Sophus::SE3d &prev_sensor_pose,
                                                           const Sophus::SE3d &sensor_pose);

    pcl::PointCloud<pcl::PointXYZ>::Ptr downsample_point_cloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud);

    pcl::PointCloud<pcl::PointXYZ>::Ptr remove_points_below_z(const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
                                                              const double z_threshold);

    pcl::PointCloud<pcl::PointXYZ>::Ptr remove_points_near_sensor(const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
                                                                  const double min_distance);

    Sophus::SE3d gicp_registration(const pcl::PointCloud<pcl::PointXYZ>::Ptr source,
                                   const pcl::PointCloud<pcl::PointXYZ>::Ptr target,
                                   const Sophus::SE3d &initial_transform_guess);

    Sophus::SE3d small_gicp_registration(const pcl::PointCloud<pcl::PointXYZ>::Ptr target,
                                         const pcl::PointCloud<pcl::PointXYZ>::Ptr source,
                                         const Sophus::SE3d &initial_transform_guess);

    void enqueue_visualization_data(const pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud,
                                    const Sophus::SE3d &sensor_pose);

    void visualization_thread_func();

    void visualize_aligned_point_clouds(const pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud, Sophus::SE3d &pose,
                                        const size_t h_res, const size_t v_res, const double view_height);

    // Sophus::SE3d predict_sensor_motion();

    pcl::VoxelGrid<pcl::PointXYZ> voxel_grid_;

    pcl::GeneralizedIterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> gicp_;
    small_gicp::RegistrationPCL<pcl::PointXYZ, pcl::PointXYZ> small_gicp_;

    double kiss_resolution_;
    kiss_matcher::KISSMatcherConfig kiss_config_;
    kiss_matcher::KISSMatcher kiss_matcher_;
    bool constrain_pose_to_se2_;

    pcl::PointCloud<pcl::PointXYZ>::Ptr prev_point_cloud_;
    pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud_;
    std::vector<Eigen::Vector3f> prev_point_cloud_vec_;

    Sophus::SE3d prev_sensor_pose_;
    Sophus::SE3d sensor_pose_;

    double z_height_threshold_;

    std::vector<Eigen::Vector3f> kiss_matcher_processed_prev_point_cloud_;
    std::vector<Eigen::Vector3f> kiss_matcher_processed_point_cloud_;

    std::vector<Eigen::Vector3f> kiss_matcher_prev_keypoints_;
    std::vector<Eigen::Vector3f> kiss_matcher_keypoints_;

    std::deque<PointCloudFrame> aligned_point_cloud_frames_;
    size_t max_point_cloud_history_;

    pcl::visualization::PCLVisualizer::Ptr viewer_;
    bool viewer_initialized_;

    std::thread viz_thread_;
    std::mutex viz_mutex_;
    std::condition_variable viz_cv_;
    bool run_viz_thread_;
    std::queue<std::shared_ptr<PointCloudFrame>> viz_queue_;
    std::deque<std::string> point_cloud_ids_;
    size_t point_cloud_counter_;
};

}  // namespace mapper3d