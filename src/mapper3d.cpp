#include "mapper3d/mapper3d.h"

#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>
#include <pcl/point_cloud.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <cmath>
#include <cstdlib>
#include <limits>
#include <small_gicp/registration/registration_helper.hpp>
#include <vector>

#include "mapper3d/common.h"
#include "mapper3d/graph_optimizer.h"
#include "mapper3d/logging.h"
#include "mapper3d/utils.h"

namespace mapper3d {

Mapper3D::Mapper3D(const Mapper3DParams& params)
    : prev_downsampled_point_cloud_(std::make_shared<pcl::PointCloud<pcl::PointXYZ>>()),
      downsampled_point_cloud_(std::make_shared<pcl::PointCloud<pcl::PointXYZ>>()),
      prev_sensor_pose_odom_frame_(Sophus::SE3d()),
      sensor_pose_odom_frame_(Sophus::SE3d()),
      sensor_pose_map_frame_(Sophus::SE3d()),
      z_height_threshold_(params.z_height_threshold),
      min_sensor_distance_(params.min_sensor_distance),
      keyframe_distance_threshold_(params.keyframe_distance_threshold),
      keyframe_rotation_threshold_(params.keyframe_rotation_threshold),
      latest_keyframe_pose_odom_frame_(Sophus::SE3d()),
      global_registration_config_(params.voxel_size),
      global_registration_(global_registration_config_),
      graph_optimizer_(params.graph_optimizer_params),
      constrain_pose_to_se2_(false),
      pose_id_(0) {
    Logger::set_global_log_level(LogLevel::Info);
    Timer::set_global_log_level(LogLevel::Info);

    // TODO: Use a more representative odometry pose uncertainty/covariance
    odometry_covariance_ = Eigen::Matrix<double, 6, 6>::Zero();
    odometry_covariance_.block<3, 3>(0, 0) = Eigen::DiagonalMatrix<double, 3>(
        params.x_odom_uncertainty, params.y_odom_uncertainty, params.z_odom_uncertainty);
    odometry_covariance_.block<3, 3>(3, 3) = Eigen::DiagonalMatrix<double, 3>(
        params.roll_odom_uncertainty, params.pitch_odom_uncertainty, params.yaw_odom_uncertainty);

    std::cout << "odometry_covariance_ = \n" << odometry_covariance_ << std::endl;

    local_registration_.setNumThreads(params.local_reg_params.num_threads);
    local_registration_.setNumNeighborsForCovariance(params.local_reg_params.num_neighbors_for_covariance);
    local_registration_.setMaxCorrespondenceDistance(params.local_reg_params.max_correspondence_distance);
    local_registration_.setVoxelResolution(params.voxel_size);
    local_registration_.setRegistrationType(params.local_reg_params.registration_type);

    voxel_grid_.setLeafSize(params.voxel_size, params.voxel_size, params.voxel_size);

    lidar_odometry_trajectory_file_ =
        utils::open_tum_file(params.lidar_odometry_trajectory_file, "Poses in odom frame");
    online_trajectory_file_ =
        utils::open_tum_file(params.online_trajectory_file, "Poses in map frame (drift corrected)");
}

void Mapper3D::spin_once(const pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud, const double timestamp) {
    Timer timer(__func__, LogLevel::InfoImportant);

    if (point_cloud->empty()) {
        return;
    }

    // point_cloud_ = point_cloud;
    downsampled_point_cloud_ = filter_point_cloud(point_cloud);

    if (prev_downsampled_point_cloud_->empty()) {
        prev_downsampled_point_cloud_ = downsampled_point_cloud_;
        return;
    }

    sensor_pose_odom_frame_ =
        estimate_lidar_odometry(prev_downsampled_point_cloud_, downsampled_point_cloud_, prev_sensor_pose_odom_frame_);

    // IMPORTANT: Only keyframes become graph nodes, but odometry stays frame-to-frame.
    if (should_add_keyframe(sensor_pose_odom_frame_)) {
        graph_optimizer_.add_frame(Frame(pose_id_, timestamp, sensor_pose_odom_frame_,
                                         pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>()),
                                         downsampled_point_cloud_, odometry_covariance_));

        latest_keyframe_pose_odom_frame_ = sensor_pose_odom_frame_;

        // IMPORTANT: pose_id_ only advances for keyframes, to stay in step with
        // GraphOptimizer::current_pose_index_. update_frame_poses() looks poses up by
        // frame id, so the two counters must not diverge.
        pose_id_++;
    }

    const Sophus::SE3d map_to_odom_transform = graph_optimizer_.map_to_odom_transform();
    // Very important: This is the drift-corrected pose after loop closure detection and graph optimization
    sensor_pose_map_frame_ = map_to_odom_transform * sensor_pose_odom_frame_;

    prev_downsampled_point_cloud_ = downsampled_point_cloud_;
    prev_sensor_pose_odom_frame_ = sensor_pose_odom_frame_;

    utils::write_tum_pose(lidar_odometry_trajectory_file_, timestamp, sensor_pose_odom_frame_);
    utils::write_tum_pose(online_trajectory_file_, timestamp, sensor_pose_map_frame_);
}

bool Mapper3D::should_add_keyframe(const Sophus::SE3d& sensor_pose) const {
    // The first frame always becomes a keyframe, to seed the graph.
    if (pose_id_ == 0) {
        return true;
    }

    const Sophus::SE3d relative_pose = latest_keyframe_pose_odom_frame_.inverse() * sensor_pose;
    const double distance = relative_pose.translation().norm();
    const double rotation = relative_pose.so3().log().norm();

    return distance >= keyframe_distance_threshold_ || rotation >= keyframe_rotation_threshold_;
}

Sophus::SE3d Mapper3D::estimate_lidar_odometry(const pcl::PointCloud<pcl::PointXYZ>::Ptr& prev_point_cloud,
                                               const pcl::PointCloud<pcl::PointXYZ>::Ptr& point_cloud,
                                               const Sophus::SE3d& prev_sensor_pose) {
    Timer timer(__func__, LogLevel::Debug);

    const auto solution = global_registration_.estimate(utils::point_cloud_to_vec(point_cloud),
                                                        utils::point_cloud_to_vec(prev_point_cloud));
    // global_registration_.print();

    const Sophus::SE3d sensor_pose_change = Sophus::SE3d(solution.rotation, solution.translation);

    // TODO: Match against the last N frames instead of just the previous one (need to be careful when using this for
    // lidar odometry vs. loop closure detection, since using last N frames for the latter won't work well with
    // small_gicp)
    const Sophus::SE3d refined_pose_change = local_registration(prev_point_cloud, point_cloud, sensor_pose_change);

    Sophus::SE3d sensor_pose = prev_sensor_pose * refined_pose_change;
    if (constrain_pose_to_se2_) {
        sensor_pose = utils::constrain_pose_to_se2(sensor_pose);
    }

    const auto global_registration_processed_point_cloud_pair = global_registration_.getProcessedInputClouds();
    global_registration_processed_prev_point_cloud_ = std::get<0>(global_registration_processed_point_cloud_pair);
    global_registration_processed_point_cloud_ = std::get<1>(global_registration_processed_point_cloud_pair);

    const auto global_registration_keypoints_pair = global_registration_.getKeypointsFromFasterPFH();
    global_registration_prev_keypoints_ = std::get<0>(global_registration_keypoints_pair);
    global_registration_keypoints_ = std::get<1>(global_registration_keypoints_pair);

    const auto global_registration_correspondences = global_registration_.getFinalCorrespondences();

    return sensor_pose;
}

Sophus::SE3d Mapper3D::local_registration(const pcl::PointCloud<pcl::PointXYZ>::Ptr& prev_point_cloud,
                                          const pcl::PointCloud<pcl::PointXYZ>::Ptr& point_cloud,
                                          const Sophus::SE3d& initial_transform_guess) {
    Timer timer(__func__, LogLevel::Info);

    local_registration_.setInputTarget(prev_point_cloud);
    local_registration_.setInputSource(point_cloud);

    pcl::PointCloud<pcl::PointXYZ> aligned_point_cloud;
    local_registration_.align(aligned_point_cloud, initial_transform_guess.matrix().cast<float>());

    // Logger::debug("Small GICP num inliers: " + std::to_string(result.num_inliers));
    // Logger::debug("Small GICP error: " + std::to_string(result.error));
    // Logger::debug("Small GICP converged: " + std::to_string(result.converged));

    Eigen::Matrix4d transformation_matrix = local_registration_.getFinalTransformation().cast<double>();
    // Orthogonalize the transformation matrix
    Sophus::SE3d refined_pose = utils::eigen_to_sophus(transformation_matrix);

    return refined_pose;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr Mapper3D::filter_point_cloud(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud) {
    Timer timer(__func__, LogLevel::Debug);

    Logger::debug("Number of points in original point cloud: " + std::to_string(point_cloud->size()));

    // Use two pointers to manage intermediate results
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in = point_cloud;
    auto cloud_out = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();

    // 1) Remove NaN points
    std::vector<int> nan_indices;
    pcl::removeNaNFromPointCloud(*cloud_in, *cloud_out, nan_indices);
    cloud_in.swap(cloud_out);
    Logger::debug("Number of points after NaN removal: " + std::to_string(cloud_in->size()));

    // 2) Remove points near the sensor
    cloud_out = remove_points_near_sensor(cloud_in, min_sensor_distance_);
    cloud_in.swap(cloud_out);
    Logger::debug("Number of points after near sensor removal: " + std::to_string(cloud_in->size()));

    // 3) Remove points below z-height threshold
    cloud_out = remove_points_below_z(cloud_in, z_height_threshold_);
    cloud_in.swap(cloud_out);
    Logger::debug("Number of points after z-height removal: " + std::to_string(cloud_in->size()));

    // 4) Downsample the remaining points
    voxel_grid_.setInputCloud(cloud_in);
    voxel_grid_.filter(*cloud_out);
    Logger::debug("Number of points after downsampling: " + std::to_string(cloud_out->size()));
    // The final result is now in cloud_out

    return cloud_out;  // Return the final filtered cloud
}

pcl::PointCloud<pcl::PointXYZ>::Ptr Mapper3D::remove_points_below_z(const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
                                                                    const double z_threshold) {
    Timer timer(__func__, LogLevel::Debug);

    auto filtered_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();

    pcl::PassThrough<pcl::PointXYZ> passthrough;
    passthrough.setInputCloud(cloud);
    passthrough.setFilterFieldName("z");                                          // Filter based on z-coordinate
    passthrough.setFilterLimits(z_threshold, std::numeric_limits<float>::max());  // Keep points with z > z_threshold
    passthrough.filter(*filtered_cloud);

    return filtered_cloud;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr Mapper3D::remove_points_near_sensor(const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
                                                                        const double min_distance) {
    Timer timer(__func__, LogLevel::Debug);

    auto filtered_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    filtered_cloud->reserve(cloud->size());

    for (const auto& point : cloud->points) {
        const double distance_squared = point.x * point.x + point.y * point.y + point.z * point.z;
        if (distance_squared > min_distance * min_distance) {
            filtered_cloud->points.push_back(point);
        }
    }

    filtered_cloud->width = filtered_cloud->points.size();
    filtered_cloud->height = 1;
    filtered_cloud->is_dense = false;

    return filtered_cloud;
}

// Sophus::SE3d Mapper3D::predict_sensor_motion() {
//     // IMPORTANT: Assumes that dt between laser scans is constant
//     // If the above assumption is true, we can use the pose change between pose(t-1) and pose(t) to predict
//     pose(t+1)

//     // Since:
//     // twist = log(pose_change)/dt
//     // and:
//     // predicted_pose = prev_pose * exp(twist * dt)
//     // we can write:
//     // predicted_pose = prev_pose * exp(log(pose_change)/dt * dt)
//     // predicted_pose = prev_pose * exp(log(pose_change))
//     // predicted_pose = prev_pose * pose_change

//     if (sensor_pose_history_.size() < 2) {
//         return Sophus::SE3d(Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero());
//     }

//     const Sophus::SE3d sensor_pose_tm2 = sensor_pose_history_[sensor_pose_history_.size() - 2];
//     const Sophus::SE3d sensor_pose_tm1 = sensor_pose_history_[sensor_pose_history_.size() - 1];
//     const Sophus::SE3d sensor_pose_change = sensor_pose_tm2.inverse() * sensor_pose_tm1;

//     return sensor_pose_change;
// }

Sophus::SE3d Mapper3D::sensor_pose_odom_frame() const { return sensor_pose_odom_frame_; }
Sophus::SE3d Mapper3D::sensor_pose_map_frame() const { return sensor_pose_map_frame_; }

const pcl::PointCloud<pcl::PointXYZ>::Ptr Mapper3D::downsampled_point_cloud() const { return downsampled_point_cloud_; }

// const std::vector<Eigen::Vector3f>& Mapper3D::point_cloud_vec() const { return point_cloud_vec_; }

const std::vector<Eigen::Vector3f>& Mapper3D::global_registration_point_cloud() const {
    return global_registration_processed_point_cloud_;
}

const std::vector<Eigen::Vector3f>& Mapper3D::global_registration_keypoints() const {
    return global_registration_keypoints_;
}

pcl::PointCloud<pcl::PointXYZ>::ConstPtr Mapper3D::get_map() { return graph_optimizer_.get_map(); }

gtsam::Values Mapper3D::get_unoptimized_poses() { return graph_optimizer_.get_unoptimized_poses(); }

gtsam::Values Mapper3D::get_optimized_poses() { return graph_optimizer_.get_optimized_poses(); }

// std::vector<Sophus::SE3d> Mapper3D::get_online_poses() { return graph_optimizer_.get_online_poses(); }

// std::vector<std::pair<size_t, size_t>> Mapper3D::get_loop_closure_edges() {
//     return graph_optimizer_.get_loop_closure_edges();
// }

gtsam::NonlinearFactorGraph Mapper3D::get_graph() { return graph_optimizer_.get_graph(); }

void Mapper3D::get_graph_visualization_data(VisualizationData& data) { graph_optimizer_.get_visualization_data(data); }

}  // namespace mapper3d