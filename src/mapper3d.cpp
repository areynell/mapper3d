#include "mapper3d/mapper3d.h"

#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>
#include <pcl/point_cloud.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <small_gicp/ann/kdtree_omp.hpp>
#include <small_gicp/factors/gicp_factor.hpp>
#include <small_gicp/registration/reduction_omp.hpp>
#include <small_gicp/registration/registration.hpp>
#include <small_gicp/registration/registration_helper.hpp>
#include <small_gicp/util/normal_estimation_omp.hpp>
#include <sophus/se3.hpp>
#include <string>
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
      prev_twist_(Sophus::SE3d::Tangent::Zero()),
      prev_point_cloud_timestamp_(-1.0),
      sensor_pose_map_frame_(Sophus::SE3d()),
      z_height_threshold_(params.z_height_threshold),
      min_sensor_distance_(params.min_sensor_distance),
      keyframe_distance_threshold_(params.keyframe_distance_threshold),
      keyframe_rotation_threshold_(params.keyframe_rotation_threshold),
      latest_keyframe_pose_odom_frame_(Sophus::SE3d()),
      global_registration_config_(params.voxel_size),
      global_registration_(global_registration_config_),
      graph_optimizer_(params.graph_optimizer_params),
      registration_num_threads_(params.local_reg_params.num_threads),
      num_neighbors_for_covariance_(params.local_reg_params.num_neighbors_for_covariance),
      wide_search_scans_after_reset_(params.local_reg_params.wide_search_scans_after_reset),
      dropped_scan_interval_(params.local_reg_params.dropped_scan_interval),
      local_map_voxel_size_(params.voxel_size),
      lru_horizon_(params.local_reg_params.lru_horizon),
      lru_clear_cycle_(params.local_reg_params.lru_clear_cycle),
      min_inlier_ratio_(params.local_reg_params.min_inlier_ratio),
      max_error_per_inlier_(params.local_reg_params.max_error_per_inlier),
      max_plausible_sensor_speed_(params.local_reg_params.max_plausible_sensor_speed),
      max_consecutive_registration_failures_(params.local_reg_params.max_consecutive_failures),
      consecutive_registration_failures_(0),
      registration_failure_count_(0),
      coasted_scan_count_(0),
      relocalization_count_(0),
      map_reset_count_(0),
      scans_since_map_reset_(0),
      point_cloud_id_(-1),  // Starts at -1 so the first scan increments it to 0 at the top of spin_once()
      keyframe_id_(0) {
    Logger::set_global_log_level(LogLevel::Info);
    Timer::set_global_log_level(LogLevel::Info);

    // TODO: Use a more representative odometry pose uncertainty/covariance
    odometry_covariance_ = Eigen::Matrix<double, 6, 6>::Zero();
    odometry_covariance_.block<3, 3>(0, 0) = Eigen::DiagonalMatrix<double, 3>(
        params.x_odom_uncertainty, params.y_odom_uncertainty, params.z_odom_uncertainty);
    odometry_covariance_.block<3, 3>(3, 3) = Eigen::DiagonalMatrix<double, 3>(
        params.roll_odom_uncertainty, params.pitch_odom_uncertainty, params.yaw_odom_uncertainty);

    std::cout << "odometry_covariance_ = \n" << odometry_covariance_ << std::endl;

    local_map_ = make_local_map();

    voxel_grid_.setLeafSize(params.voxel_size, params.voxel_size, params.voxel_size);

    lidar_odometry_trajectory_file_ =
        utils::open_tum_file(params.lidar_odometry_trajectory_file, "Poses in odom frame");
    online_trajectory_file_ =
        utils::open_tum_file(params.online_trajectory_file, "Poses in map frame (drift corrected)");
}

void Mapper3D::check_homogeneous_point_format(const pcl::PointCloud<pcl::PointXYZ>::Ptr& point_cloud) const {
    // Checking first scan only. A point's fourth number, data[3], is padding that PCL fills with
    // 1.0 by convention, which makes it the w of a homogeneous coordinate.
    // local_map_->insert multiplies each point by the 4x4 sensor pose, and only w = 1.0 applies
    // that pose multiplication correctly.
    if (point_cloud_id_ != 0 || point_cloud->empty()) {
        return;
    }

    const float w = point_cloud->points.front().data[3];
    if (w != 1.0f) {
        Logger::error("Filtered points have data[3] = " + std::to_string(w) +
                      ", not 1.0. Every point inserted into the local map will be mistransformed.");
    }
}

void Mapper3D::spin_once(const pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud, const double point_cloud_timestamp) {
    Timer timer(__func__, LogLevel::InfoImportant);

    // IMPORTANT: Incremented at the start of spin_once, so point_cloud_id_ is always the current point cloud's ID even
    // if there is an early return.
    ++point_cloud_id_;

    if (point_cloud->empty()) {
        return;
    }

    check_homogeneous_point_format(point_cloud);

    // point_cloud_ = point_cloud;
    downsampled_point_cloud_ = filter_point_cloud(point_cloud);

    const double dt = prev_point_cloud_timestamp_ < 0.0 ? 0.0 : point_cloud_timestamp - prev_point_cloud_timestamp_;

    sensor_pose_odom_frame_ =
        estimate_lidar_odometry(prev_sensor_pose_odom_frame_, prev_twist_, downsampled_point_cloud_, dt);

    // IMPORTANT: Only keyframes become graph nodes. Odometry runs on every scan.
    if (should_add_keyframe(sensor_pose_odom_frame_)) {
        graph_optimizer_.add_frame(Frame(keyframe_id_, point_cloud_id_, point_cloud_timestamp, sensor_pose_odom_frame_,
                                         pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>()),
                                         downsampled_point_cloud_, odometry_covariance_));

        latest_keyframe_pose_odom_frame_ = sensor_pose_odom_frame_;

        keyframe_id_++;
    }

    const Sophus::SE3d map_to_odom_transform = graph_optimizer_.map_to_odom_transform();
    // VERY IMPORTANT: This is the drift-corrected pose after loop closure detection and graph optimization
    sensor_pose_map_frame_ = map_to_odom_transform * sensor_pose_odom_frame_;

    utils::write_tum_pose(lidar_odometry_trajectory_file_, point_cloud_timestamp, sensor_pose_odom_frame_);
    utils::write_tum_pose(online_trajectory_file_, point_cloud_timestamp, sensor_pose_map_frame_);

    // IMPORTANT: The estimate_twist() call must come first, because it reads prev_sensor_pose_odom_frame_ as the older
    // of the two poses and the line below it overwrites that member.
    prev_twist_ = estimate_twist(prev_sensor_pose_odom_frame_, sensor_pose_odom_frame_, dt);
    prev_sensor_pose_odom_frame_ = sensor_pose_odom_frame_;
    prev_point_cloud_timestamp_ = point_cloud_timestamp;
    prev_downsampled_point_cloud_ = downsampled_point_cloud_;
}

bool Mapper3D::should_add_keyframe(const Sophus::SE3d& sensor_pose) const {
    // The first frame always becomes a keyframe.
    if (keyframe_id_ == 0) {
        return true;
    }

    const Sophus::SE3d relative_pose = latest_keyframe_pose_odom_frame_.inverse() * sensor_pose;
    const double distance = relative_pose.translation().norm();
    const double rotation = relative_pose.so3().log().norm();

    return distance >= keyframe_distance_threshold_ || rotation >= keyframe_rotation_threshold_;
}

Sophus::SE3d::Tangent Mapper3D::estimate_twist(const Sophus::SE3d& prev_pose, const Sophus::SE3d& pose,
                                               const double dt) {
    // Two poses measured over a real interval are needed before there is any velocity. The interval is
    // zero until a second point cloud has arrived, so the first scan reports no movement.
    if (dt <= 0.0) {
        return Sophus::SE3d::Tangent::Zero();
    }

    const Sophus::SE3d relative_pose = prev_pose.inverse() * pose;
    const Sophus::SE3d::Tangent twist = relative_pose.log() / dt;
    return twist;
}

Sophus::SE3d Mapper3D::estimate_lidar_odometry(const Sophus::SE3d& prev_pose, const Sophus::SE3d::Tangent& prev_twist,
                                               const pcl::PointCloud<pcl::PointXYZ>::Ptr& point_cloud,
                                               const double dt) {
    Timer timer(__func__, LogLevel::Info);

    small_gicp::PointCloudProxy<pcl::PointXYZ> source(*point_cloud, point_cloud_covariances_);
    estimate_point_cloud_covariances(source);

    // The first scan defines the odom frame. The local map is empty, so there is nothing to register
    // against: put this scan in as the map's first contents and set its pose to the origin.
    if (local_map_->size() == 0) {
        local_map_->insert(source);
        return Sophus::SE3d();
    }

    const Sophus::SE3d predicted_sensor_pose = predict_sensor_pose(prev_pose, prev_twist, dt);

    const small_gicp::RegistrationResult registration_result = align_to_local_map(source, predicted_sensor_pose, dt);

    Logger::debug("Local registration: converged = " + std::to_string(registration_result.converged) +
                  " iterations = " + std::to_string(registration_result.iterations) +
                  " inliers = " + std::to_string(registration_result.num_inliers) + "/" +
                  std::to_string(point_cloud->size()) + " error = " + std::to_string(registration_result.error) +
                  " map_voxels = " + std::to_string(local_map_->size()));

    scans_since_map_reset_++;

    if (registration_is_acceptable(registration_result, point_cloud->size())) {
        consecutive_registration_failures_ = 0;
        const Sophus::SE3d sensor_pose = utils::eigen_to_sophus(registration_result.T_target_source.matrix());
        local_map_->insert(source, utils::sophus_to_isometry(sensor_pose));
        return sensor_pose;
    } else {
        registration_failure_count_++;
        consecutive_registration_failures_++;

        Logger::warn("Registration failed: inlier ratio " +
                     std::to_string(inlier_ratio(registration_result, point_cloud->size())) + " (min " +
                     std::to_string(min_inlier_ratio_) + "), error per inlier " +
                     std::to_string(error_per_inlier(registration_result)) + " (max " +
                     std::to_string(max_error_per_inlier_) + "). Failure " +
                     std::to_string(consecutive_registration_failures_) + " in a row, " +
                     std::to_string(registration_failure_count_) + " this run.");

        // Enough failures in a row means that the front-end lidar odometry is lost, so we can no longer safely use pose
        // prediction from estimated twist, and instead need to perform a recovery using global registration. However,
        // if global registration also fails, we fall back to using the predicted pose once again and retry global
        // registration when the next point cloud arrives.
        if (consecutive_registration_failures_ >= max_consecutive_registration_failures_) {
            const std::optional<Sophus::SE3d> recovered_sensor_pose =
                recover_with_global_registration(prev_pose, point_cloud, source, dt);
            if (recovered_sensor_pose.has_value()) {
                return recovered_sensor_pose.value();
            }
        }

        // If registration fails, use the twist-based predicted pose.
        // IMPORTANT: The point cloud is NOT added to the local map. Its pose is only a guess, and adding it would
        // likely contaminate the local map against which newer point clouds are registered.
        coasted_scan_count_++;
        Logger::warn(
            "Using constant-velocity pose prediction. Point cloud data will NOT be added to the local map. "
            "Registration has failed " +
            std::to_string(consecutive_registration_failures_) + " times in a row. " +
            std::to_string(coasted_scan_count_) + " scans have used the predicted pose this run.");

        return predicted_sensor_pose;
    }
}

Sophus::SE3d Mapper3D::predict_sensor_pose(const Sophus::SE3d& pose, const Sophus::SE3d::Tangent& twist,
                                           const double dt) const {
    return pose * Sophus::SE3d::exp(twist * dt);
}

void Mapper3D::estimate_point_cloud_covariances(small_gicp::PointCloudProxy<pcl::PointXYZ>& source) {
    // Fills point_cloud_covariances_ for every point of the source cloud.

    small_gicp::KdTreeBuilderOMP builder(registration_num_threads_);
    small_gicp::UnsafeKdTree<small_gicp::PointCloudProxy<pcl::PointXYZ>> tree(source, builder);
    small_gicp::estimate_covariances_omp(source, tree, num_neighbors_for_covariance_, registration_num_threads_);
}

small_gicp::RegistrationResult Mapper3D::align_to_local_map(const small_gicp::PointCloudProxy<pcl::PointXYZ>& source,
                                                            const Sophus::SE3d& initial_guess, const double dt) const {
    if (use_wide_correspondence_search(dt)) {
        // Search the point's own voxel and its 26 neighbours (3 x 3 x 3 cube)
        local_map_->set_search_offsets(27);
    } else {
        // Search the point's own voxel only
        local_map_->set_search_offsets(1);
    }

    small_gicp::Registration<small_gicp::GICPFactor, small_gicp::ParallelReductionOMP> registration;
    registration.reduction.num_threads = registration_num_threads_;
    // IMPORTANT: DISABLE the distance rejector. The search offsets above already limit how far away a match can be.
    // The default of 1 m would reject good matches from the 27-voxel search.
    registration.rejector.max_dist_sq = std::numeric_limits<double>::max();

    return registration.align(*local_map_, source, *local_map_, utils::sophus_to_isometry(initial_guess));
}

bool Mapper3D::use_wide_correspondence_search(const double dt) const {
    // Search wide while the local map is new.
    if (scans_since_map_reset_ < wide_search_scans_after_reset_) {
        return true;
    }

    // Search wide if we have a dropped scan
    if (dt > dropped_scan_interval_) {
        Logger::warn("Scans have likely been dropped. Will use wide correspondence search.");
        return true;
    }

    return false;
}

small_gicp::GaussianVoxelMap::Ptr Mapper3D::make_local_map() const {
    auto local_map = std::make_shared<small_gicp::GaussianVoxelMap>(local_map_voxel_size_);
    local_map->lru_horizon = lru_horizon_;
    local_map->lru_clear_cycle = lru_clear_cycle_;
    return local_map;
}

void Mapper3D::reset_local_map(const Sophus::SE3d& pose, const small_gicp::PointCloudProxy<pcl::PointXYZ>& source) {
    // GaussianVoxelMap has no clear(), so the whole object is replaced.
    local_map_ = make_local_map();
    local_map_->insert(source, utils::sophus_to_isometry(pose));

    // A rebuilt map is as unproven as the first one, so open the wide search window again.
    scans_since_map_reset_ = 0;
}

std::optional<Sophus::SE3d> Mapper3D::global_registration_pose(const Sophus::SE3d& prev_pose,
                                                               const pcl::PointCloud<pcl::PointXYZ>::Ptr& point_cloud,
                                                               const double dt) {
    Timer timer(__func__, LogLevel::InfoImportant);

    if (prev_downsampled_point_cloud_->empty()) {
        Logger::warn("Global registration: there is no previous point cloud to register against.");
        return std::nullopt;
    }

    const auto solution = global_registration_.estimate(utils::point_cloud_to_vec(point_cloud),
                                                        utils::point_cloud_to_vec(prev_downsampled_point_cloud_));
    if (!solution.valid) {
        Logger::warn("Global registration: no valid solution.");
        return std::nullopt;
    }

    const Sophus::SE3d pose_increment(solution.rotation, solution.translation);

    // Throw out grossly incorrect answers a global solver can return.
    const double motion = pose_increment.translation().norm();
    const double motion_limit = max_plausible_sensor_speed_ * dt;
    if (motion > motion_limit) {
        Logger::warn("Global registration: the solution implies " + std::to_string(motion) + " m of movement over " +
                     std::to_string(dt) + " s, over the " + std::to_string(motion_limit) + " m limit. Rejected.");
        return std::nullopt;
    }

    return prev_pose * pose_increment;
}

std::optional<Sophus::SE3d> Mapper3D::recover_with_global_registration(
    const Sophus::SE3d& prev_pose, const pcl::PointCloud<pcl::PointXYZ>::Ptr& point_cloud,
    const small_gicp::PointCloudProxy<pcl::PointXYZ>& source, const double dt) {
    Timer timer(__func__, LogLevel::InfoImportant);

    Logger::info_important("Local registration failed " + std::to_string(consecutive_registration_failures_) +
                           " times in a row. Trying global registration against the previous point cloud.");

    const std::optional<Sophus::SE3d> recovered_sensor_pose = global_registration_pose(prev_pose, point_cloud, dt);
    if (!recovered_sensor_pose.has_value()) {
        return std::nullopt;
    }

    // First choice: perform local registration against the existing local map again, using the pose obtained from
    // global registration as the initial guess pose. If this passes, the local map is kept.
    const small_gicp::RegistrationResult registration_result =
        align_to_local_map(source, recovered_sensor_pose.value(), dt);

    if (registration_is_acceptable(registration_result, point_cloud->size())) {
        relocalization_count_++;

        Logger::info_important(
            "Re-registration against local map using initial pose from global registration passed. Keeping the local "
            "map (kept " +
            std::to_string(relocalization_count_) + " times this run).");

        consecutive_registration_failures_ = 0;

        const Sophus::SE3d sensor_pose = utils::eigen_to_sophus(registration_result.T_target_source.matrix());
        local_map_->insert(source, utils::sophus_to_isometry(sensor_pose));
        return sensor_pose;
    }

    Logger::warn(
        "Re-registration against local map using initial pose from global registration failed. Rebuilding the "
        "local map from this point cloud.");

    // Second choice: rebuild the local map from this point cloud, placed at the global registration pose.
    reset_local_map(recovered_sensor_pose.value(), source);
    consecutive_registration_failures_ = 0;
    map_reset_count_++;

    return recovered_sensor_pose.value();
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

bool Mapper3D::registration_is_acceptable(const small_gicp::RegistrationResult& result, const size_t num_points) const {
    // No points, or no matches at all, is a failure.
    if (num_points == 0 || result.num_inliers == 0) {
        return false;
    }

    if (inlier_ratio(result, num_points) >= min_inlier_ratio_ && error_per_inlier(result) <= max_error_per_inlier_) {
        return true;
    }
    return false;
}

double Mapper3D::inlier_ratio(const small_gicp::RegistrationResult& result, const size_t num_points) const {
    if (num_points == 0) {
        return 0.0;
    }
    return static_cast<double>(result.num_inliers) / static_cast<double>(num_points);
}

double Mapper3D::error_per_inlier(const small_gicp::RegistrationResult& result) const {
    if (result.num_inliers == 0) {
        return 0.0;
    }
    return result.error / static_cast<double>(result.num_inliers);
}

Sophus::SE3d Mapper3D::sensor_pose_odom_frame() const { return sensor_pose_odom_frame_; }
Sophus::SE3d Mapper3D::sensor_pose_map_frame() const { return sensor_pose_map_frame_; }

const pcl::PointCloud<pcl::PointXYZ>::Ptr Mapper3D::downsampled_point_cloud() const { return downsampled_point_cloud_; }

pcl::PointCloud<pcl::PointXYZ>::ConstPtr Mapper3D::get_map() { return graph_optimizer_.get_map(); }

gtsam::Values Mapper3D::get_unoptimized_poses() { return graph_optimizer_.get_unoptimized_poses(); }

gtsam::Values Mapper3D::get_optimized_poses() { return graph_optimizer_.get_optimized_poses(); }

gtsam::NonlinearFactorGraph Mapper3D::get_graph() { return graph_optimizer_.get_graph(); }

void Mapper3D::get_graph_visualization_data(VisualizationData& data) { graph_optimizer_.get_visualization_data(data); }

}  // namespace mapper3d