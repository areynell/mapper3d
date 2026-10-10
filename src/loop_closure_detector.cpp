#include "mapper3d/loop_closure_detector.h"

#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <cassert>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

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

    scan_context_manager_.makeAndSaveScancontextAndKeys(*frame.downsampled_point_cloud);

    std::pair<int, float> nearest_keyframe_id_and_relative_yaw = scan_context_manager_.detectLoopClosureID();
    const int nearest_keyframe_id = nearest_keyframe_id_and_relative_yaw.first;
    const float relative_yaw = nearest_keyframe_id_and_relative_yaw.second;
    if (nearest_keyframe_id == -1) {
        return std::nullopt;  // No loop closure found
    }

    const Frame& nearest_frame = frames_.at(nearest_keyframe_id);

    // TODO: Is it possible to call estimate directly on FasterPFH (keypoint, descriptor) pairs?
    const auto solution =
        global_registration_.estimate(utils::point_cloud_to_vec(frame.downsampled_point_cloud),
                                      utils::point_cloud_to_vec(nearest_frame.downsampled_point_cloud));
    const size_t num_final_inliers = global_registration_.getNumFinalInliers();
    if (num_final_inliers < min_final_inliers_) {
        Logger::info("Rejected loop closure between keyframes " + std::to_string(nearest_keyframe_id) + " and " +
                     std::to_string(frame.keyframe_id) + ": KISS-Matcher final inliers " +
                     std::to_string(num_final_inliers) + " (min " + std::to_string(min_final_inliers_) + ")");
        return std::nullopt;
    }

    const Sophus::SE3d relative_pose(solution.rotation, solution.translation);
    const small_gicp::RegistrationResult registration_result =
        local_registration(nearest_frame.downsampled_point_cloud, frame.downsampled_point_cloud, relative_pose);
    const double inlier_ratio =
        utils::inlier_ratio(registration_result.num_inliers, frame.downsampled_point_cloud->size());
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

    // Logger::debug("Small GICP num inliers: " + std::to_string(result.num_inliers));
    // Logger::debug("Small GICP error: " + std::to_string(result.error));
    // Logger::debug("Small GICP converged: " + std::to_string(result.converged));

    return local_registration_.getRegistrationResult();
}

void LoopClosureDetector::update_submaps(const Frame& frame) {
    if (!current_submap_.initialized()) {
        current_submap_ = Submap(frame.keyframe_id, frame.pose, frame.downsampled_point_cloud);
    }
    current_submap_.add_point_cloud(frame.keyframe_id, frame.pose, frame.downsampled_point_cloud);

    // If the submap is finalized, add it to submaps_ and start a new submap
    if (current_submap_.finalized()) {
        submaps_.emplace(current_submap_.id, current_submap_);
        current_submap_ = Submap(frame.keyframe_id, frame.pose, frame.downsampled_point_cloud);
    }
}

void LoopClosureDetector::visualize_point_cloud_alignment(const Submap& prev_submap, const Frame& frame,
                                                          const Sophus::SE3d& relative_pose, const size_t image_width,
                                                          const size_t image_height, const float view_range_x,
                                                          const float view_range_y) {
    // Transform previous point cloud from sensor frame to world frame
    const pcl::PointCloud<pcl::PointXYZ>::Ptr prev_point_cloud =
        utils::point_vec_to_cloud(prev_submap.downsampled_point_cloud_vec);
    auto transformed_prev_point_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    pcl::transformPointCloud(*prev_point_cloud, *transformed_prev_point_cloud,
                             prev_submap.origin_pose.matrix().cast<double>());

    // Transform current point cloud from sensor frame to world frame
    const pcl::PointCloud<pcl::PointXYZ>::Ptr current_point_cloud = frame.downsampled_point_cloud;
    const Sophus::SE3d corrected_sensor_pose = prev_submap.origin_pose * relative_pose;
    pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_point_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    pcl::transformPointCloud(*current_point_cloud, *transformed_point_cloud,
                             corrected_sensor_pose.matrix().cast<double>());

    // Get the current frame's position to center the view
    const Eigen::Vector3d current_frame_position = frame.pose.translation();
    const float center_x = static_cast<float>(current_frame_position.x());
    const float center_y = static_cast<float>(current_frame_position.y());

    // Calculate world coordinate bounds centered on the current frame's pose
    const float x_min = center_x - view_range_x / 2.0f;
    const float x_max = center_x + view_range_x / 2.0f;
    const float y_min = center_y - view_range_y / 2.0f;
    const float y_max = center_y + view_range_y / 2.0f;

    // Ensure (x_max - x_min) and (y_max - y_min) are not zero to avoid division by zero
    if (x_max <= x_min || y_max <= y_min) {
        std::cerr << "Error: Invalid world bounds for visualization (x_max <= x_min or y_max <= y_min)." << std::endl;
        return;
    }
    // Calculate scaling factors based on the chosen fixed bounds
    const float x_scale = static_cast<float>(image_width) / (x_max - x_min);
    const float y_scale = static_cast<float>(image_height) / (y_max - y_min);

    // Function to map a 3D point (using X and Y for top-down) to a 2D pixel coordinate
    auto map_point_to_pixel = [&](const pcl::PointXYZ& point) -> cv::Point {
        const int px = static_cast<int>((point.x - x_min) * x_scale);
        // Flip y-axis (image coordinates have origin at top-left, world Y might be up or forward)
        const int py = image_height - 1 - static_cast<int>((point.y - y_min) * y_scale);
        return cv::Point(px, py);
    };

    cv::Mat loop_closure_image(image_height, image_width, CV_8UC3, cv::Scalar(0, 0, 0));

    // Draw previous point cloud
    for (const auto& point : transformed_prev_point_cloud->points) {
        cv::Point pixel = map_point_to_pixel(point);
        if (pixel.x >= 0 && pixel.x < image_width && pixel.y >= 0 && pixel.y < image_height) {
            cv::circle(loop_closure_image, pixel, 2, cv::Scalar(0, 0, 255), -1);
        }
    }

    // Draw current point cloud
    for (const auto& point : transformed_point_cloud->points) {
        cv::Point pixel = map_point_to_pixel(point);
        if (pixel.x >= 0 && pixel.x < image_width && pixel.y >= 0 && pixel.y < image_height) {
            cv::circle(loop_closure_image, pixel, 2, cv::Scalar(0, 255, 0), -1);
        }
    }

    // const std::string window_name = "Top-Down Aligned Point Clouds";
    // cv::imshow(window_name, image);
    // cv::waitKey(0);                  // Wait for a key press to close the window
    // cv::destroyWindow(window_name);  // Clean up the window

    // Save image to disk
    std::string filename =
        std::to_string(prev_submap.id) + "_to_" + std::to_string(frame.keyframe_id) + "_loop_closure.png";
    cv::imwrite(filename, loop_closure_image);
    Logger::debug("Saved loop closure image: " + filename);
}

}  // namespace mapper3d