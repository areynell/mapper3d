#pragma once

#include <pcl/filters/voxel_grid.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/registration/gicp.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <kiss_matcher/FasterPFH.hpp>
#include <kiss_matcher/KISSMatcher.hpp>
#include <sophus/se3.hpp>

struct Mapper3DParams {
    int max_iterations = 100;
    double max_correspondence_distance = 0.05;  // This is very high, need to use motion prediction to lower this
                                                // (i.e. get a good initial guess before starting ICP)
    double transformation_epsilon = 1e-4;
    double euclidean_fitness_epsilon = 1e-6;
    double ransac_outlier_rejection_threshold = 0.05;
    double ransac_iterations = 100;

    float voxel_size = 0.2F;  // Voxel size for downsampling (m)
};

class Mapper3D {
   public:
    Mapper3D() = delete;
    Mapper3D(const Mapper3DParams &params);

    void spin_once(const pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud);

    Sophus::SE3d sensor_pose() const;

    Sophus::SE3d eigen_to_sophus(const Eigen::Matrix4d &eigen_matrix);

   private:
    pcl::PointCloud<pcl::PointXYZ>::Ptr downsample_point_cloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud);

    pcl::PointCloud<pcl::PointXYZ>::Ptr remove_ground_plane(const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);

    pcl::PointCloud<pcl::PointXYZ>::Ptr remove_points_below_z(const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
                                                              const double z_threshold);

    std::vector<Eigen::Vector3f> convert_point_cloud_to_vec(const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);

    Sophus::SE3d gicp_registration(const pcl::PointCloud<pcl::PointXYZ>::Ptr source,
                                   const pcl::PointCloud<pcl::PointXYZ>::Ptr target);

    void visualize_point_cloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud);

    void add_keypoints_to_visualizer(const pcl::PointCloud<pcl::PointXYZ>::Ptr keypoints);

    // pcl::PointCloud<PointXYZT>::Ptr dewarp_point_cloud(const pcl::PointCloud<PointXYZT>::Ptr &point_cloud,
    //                                                    const Eigen::Matrix4f &prev_sensor_pose,
    //                                                    const Eigen::Matrix4f &sensor_pose);

    // Sophus::SE3d predict_sensor_motion();

    pcl::VoxelGrid<pcl::PointXYZ> voxel_grid_;

    pcl::GeneralizedIterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> gicp_;

    double kiss_resolution_;
    kiss_matcher::KISSMatcherConfig kiss_config_;
    kiss_matcher::KISSMatcher kiss_matcher_;

    pcl::PointCloud<pcl::PointXYZ>::Ptr prev_point_cloud_;
    std::vector<Eigen::Vector3f> prev_point_cloud_vec_;

    Sophus::SE3d prev_sensor_pose_;
    Sophus::SE3d sensor_pose_;

    pcl::visualization::PCLVisualizer::Ptr viewer_;
    bool viewer_initialized_;

    double z_height_threshold_;
};