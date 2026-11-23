#include "mapper3d/loop_closure_detector.h"

#include <iostream>

#include "mapper3d/common.h"
#include "mapper3d/logging.h"
#include "mapper3d/utils.h"

namespace mapper3d {

LoopClosureDetector::LoopClosureDetector(const LoopClosureDetectorParams& params)
    : global_registration_config_(params.voxel_size),
      global_registration_(global_registration_config_),
      min_final_inliers_(params.min_final_inliers),
      min_inlier_ratio_(params.min_inlier_ratio) {
    local_registration_.setMaxCorrespondenceDistance(params.max_correspondence_distance);
    local_registration_.setNumThreads(params.num_threads);

    loop_closure_covariance_ = utils::gtsam_pose_covariance(
        Eigen::Vector3d(params.roll_loop_uncertainty, params.pitch_loop_uncertainty, params.yaw_loop_uncertainty),
        Eigen::Vector3d(params.x_loop_uncertainty, params.y_loop_uncertainty, params.z_loop_uncertainty));

    std::cout << "loop_closure_covariance_ = " << std::endl << loop_closure_covariance_ << std::endl;
}

std::optional<LoopClosure> LoopClosureDetector::detect_loop_closures(const Frame& frame) {
    Timer timer(__func__, LogLevel::Debug);

    // TODO: Match submaps instead of single point clouds to get better loop closure detection
    // TODO: Should the poses in frames_ be updated after graph optimization?
    frames_.emplace(frame.keyframe_id, frame);  // Store the current frame for future reference

    scan_context_manager_.makeAndSaveScancontextAndKeys(*frame.point_cloud);

    std::pair<int, float> nearest_keyframe_id_and_relative_yaw = scan_context_manager_.detectLoopClosureID();
    const int nearest_keyframe_id = nearest_keyframe_id_and_relative_yaw.first;
    const float relative_yaw = nearest_keyframe_id_and_relative_yaw.second;
    if (nearest_keyframe_id == -1) {
        return std::nullopt;  // No loop closure found
    }

    const Frame& nearest_frame = frames_.at(nearest_keyframe_id);

    // TODO: Is it possible to call estimate directly on FasterPFH (keypoint, descriptor) pairs?
    const auto solution = global_registration_.estimate(utils::point_cloud_to_vec(frame.point_cloud),
                                                        utils::point_cloud_to_vec(nearest_frame.point_cloud));
    const size_t num_final_inliers = global_registration_.getNumFinalInliers();
    if (num_final_inliers < min_final_inliers_) {
        Logger::info("Rejected loop closure between keyframes " + std::to_string(nearest_keyframe_id) + " and " +
                     std::to_string(frame.keyframe_id) + ": KISS-Matcher final inliers " +
                     std::to_string(num_final_inliers) + " (min " + std::to_string(min_final_inliers_) + ")");
        return std::nullopt;
    }

    const Sophus::SE3d relative_pose(solution.rotation, solution.translation);
    const small_gicp::RegistrationResult registration_result =
        local_registration(nearest_frame.point_cloud, frame.point_cloud, relative_pose);
    const double inlier_ratio = utils::inlier_ratio(registration_result.num_inliers, frame.point_cloud->size());
    if (inlier_ratio < min_inlier_ratio_) {
        Logger::info("Rejected loop closure between keyframes " + std::to_string(nearest_keyframe_id) + " and " +
                     std::to_string(frame.keyframe_id) + ": GICP inlier ratio " + std::to_string(inlier_ratio) +
                     " (min " + std::to_string(min_inlier_ratio_) + ")");
        return std::nullopt;
    }

    const Sophus::SE3d refined_relative_pose = utils::eigen_to_sophus(registration_result.T_target_source.matrix());

    const auto loop_closure =
        LoopClosure(nearest_keyframe_id, frame.keyframe_id, refined_relative_pose, loop_closure_covariance_);

    Logger::info_important("Found loop closure between keyframes " + std::to_string(loop_closure.prev_keyframe_id) +
                           " and " + std::to_string(loop_closure.keyframe_id));

    return loop_closure;
}

small_gicp::RegistrationResult LoopClosureDetector::local_registration(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr prev_point_cloud, const pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud,
    const Sophus::SE3d& initial_transform_guess) {
    Timer timer(__func__, LogLevel::Info);

    local_registration_.setInputTarget(prev_point_cloud);
    local_registration_.setInputSource(point_cloud);

    pcl::PointCloud<pcl::PointXYZ> aligned_point_cloud;
    local_registration_.align(aligned_point_cloud, initial_transform_guess.matrix().cast<float>());

    return local_registration_.getRegistrationResult();
}

}  // namespace mapper3d
