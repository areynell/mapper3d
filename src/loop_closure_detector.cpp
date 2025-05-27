#include "mapper3d/loop_closure_detector.h"

#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <cassert>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "mapper3d/common.h"
#include "mapper3d/utils.h"

namespace mapper3d {

LoopClosureDetector::LoopClosureDetector(const LoopClosureDetectorParams& params)
    : kiss_config_(params.voxel_size),
      kiss_matcher_(kiss_config_),
      distance_threshold_(params.distance_threshold),
      num_inliers_threshold_(params.num_inliers_threshold),
      num_prev_frames_to_skip_(params.num_prev_frames_to_skip) {
    // TODO: Use a more representative odometry pose uncertainty/covariance
    loop_closure_covariance_ = Eigen::Matrix<double, 6, 6>::Zero();
    loop_closure_covariance_.block<3, 3>(0, 0) = Eigen::DiagonalMatrix<double, 3>(
        params.x_loop_uncertainty, params.y_loop_uncertainty, params.z_loop_uncertainty);
    loop_closure_covariance_.block<3, 3>(3, 3) = Eigen::DiagonalMatrix<double, 3>(
        params.roll_loop_uncertainty, params.pitch_loop_uncertainty, params.yaw_loop_uncertainty);

    std::cout << " loop_closure_covariance_ = " << std::endl << loop_closure_covariance_ << std::endl;
}

std::optional<LoopClosure> LoopClosureDetector::detect_loop_closures(const Frame& frame) {
    Timer timer("LoopClosureDetector::detect_loop_closures");

    // clang-format off
    // Loop closure detection logic is as follows:
    // 1) Compare the current frame with all previous frames
    // 2) If the current frame has a position difference < threshold with a previous frame, then try and match the two frames using KISS-Matcher global registration
    // 3) If the two frames match within a certain threshold, then perform a GICP registration to refine the match
    // 4) If the GICP registration is successful, add the loop closure to the graph optimizer
    // 5) Return the loop closure
    // clang-format on

    update_submaps(frame);

    size_t max_num_inliers = 0;
    bool loop_closure_found = false;
    Submap best_submap;  // Only used for visualization
    LoopClosure best_loop_closure;

    for (const auto& [id, submap] : submaps_) {
        if (frame.id == id) {
            continue;  // Skip the same frame
        }
        // Check that current frame isn't matched with a previous frame that is too close in time
        if ((frame.id - id) < num_prev_frames_to_skip_) {
            continue;  // Skip frames that are too close
        }

        // Check if the distance between the two frames is below a certain threshold
        const double distance = (frame.pose.translation() - submap.origin_pose.translation()).norm();
        assert(distance >= 0.0);
        if (distance < distance_threshold_) {
            // TODO: Is it possible to call estimate directly on FasterPFH (keypoint, descriptor) pairs?
            // Perform KISS-Matcher global registration

            // TODO: Optimize this by first computing FasterPFH features and storing them in the submap
            const auto solution = kiss_matcher_.estimate(frame.point_cloud, submap.downsampled_point_cloud_vec);
            std::cout << "kiss_matcher_.getNumRotationInliers() = " << kiss_matcher_.getNumRotationInliers()
                      << std::endl;
            std::cout << "kiss_matcher_.getNumFinalInliers() = " << kiss_matcher_.getNumFinalInliers()
                      << ", num_inliers_threshold_ = " << num_inliers_threshold_ << std::endl;

            // QUESTION: Why do we generally get such low inlier counts?
            if (kiss_matcher_.getNumFinalInliers() > num_inliers_threshold_ &&
                kiss_matcher_.getNumFinalInliers() > max_num_inliers) {
                loop_closure_found = true;
                max_num_inliers = kiss_matcher_.getNumFinalInliers();
                best_submap = submap;  // Store the best previous frame for visualization

                // Find the closest pose in the submap to the current frame's pose
                const auto [closest_pose_id, closest_pose_in_submap] = submap.find_closest_pose(frame.pose);

                // Compute relative pose from closest pose in submap to the current frame's pose
                const Sophus::SE3d relative_pose_to_submap_origin(solution.rotation, solution.translation);
                const Sophus::SE3d relative_pose =
                    closest_pose_in_submap.inverse() * (submap.origin_pose * relative_pose_to_submap_origin);

                best_loop_closure = LoopClosure(closest_pose_id, frame.id, relative_pose, loop_closure_covariance_);

                std::cout << "\033[33m" << "Found loop closure between frames " << closest_pose_id << " and "
                          << frame.id << " with " << max_num_inliers << " inliers." << "\033[0m" << std::endl;
            }
        }
    }

    if (loop_closure_found) {
        std::cout << "\033[32m" << "Found loop closure between frames " << best_loop_closure.prev_id << " and "
                  << best_loop_closure.id << " with " << max_num_inliers << " inliers." << "\033[0m" << std::endl;

        visualize_point_cloud_alignment(best_submap, frame, best_loop_closure.relative_pose, 1920, 1080, 200.0F,
                                        200.0F);

        // // Perform GICP local registration to refine the match
        // const Sophus::SE3d refined_pose_change =
        //     small_gicp_registration(point_cloud_vec, prev_point_cloud_vec, solution.relative_pose);
        // if (refined_pose_change.is_valid()) {
        //     return LoopClosure(prev_id, frame.id, refined_pose_change, Eigen::Matrix<double, 6,
        //     6>::Identity());
        // }

        return best_loop_closure;
    }

    std::cout << "\033[31m" << "No loop closures found for current frame " << "\033[0m" << std::endl;
    return std::nullopt;  // No loop closure found
}

void LoopClosureDetector::update_submaps(const Frame& frame) {
    if (!current_submap_.initialized()) {
        current_submap_ = Submap(frame.id, frame.pose, utils::point_vec_to_cloud(frame.point_cloud));
    }
    current_submap_.add_point_cloud(frame.id, frame.pose, utils::point_vec_to_cloud(frame.point_cloud));

    // If the submap is finalized, add it to submaps_ and start a new submap
    if (current_submap_.finalized()) {
        submaps_.emplace(current_submap_.id, current_submap_);
        current_submap_ = Submap(frame.id, frame.pose, utils::point_vec_to_cloud(frame.point_cloud));
    }
}

void LoopClosureDetector::visualize_point_cloud_alignment(const Submap& prev_submap, const Frame& frame,
                                                          const Sophus::SE3d& relative_pose, const size_t image_width,
                                                          const size_t image_height, const float view_range_x,
                                                          const float view_range_y) {
    // Transform previous point cloud from sensor frame to world frame
    const pcl::PointCloud<pcl::PointXYZ>::Ptr prev_point_cloud =
        utils::point_vec_to_cloud(prev_submap.downsampled_point_cloud_vec);
    pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_prev_point_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::transformPointCloud(*prev_point_cloud, *transformed_prev_point_cloud,
                             prev_submap.origin_pose.matrix().cast<double>());

    // Transform current point cloud from sensor frame to world frame
    const pcl::PointCloud<pcl::PointXYZ>::Ptr current_point_cloud = utils::point_vec_to_cloud(frame.point_cloud);
    const Sophus::SE3d corrected_sensor_pose = prev_submap.origin_pose * relative_pose;
    pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_point_cloud(new pcl::PointCloud<pcl::PointXYZ>);
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
    std::string filename = std::to_string(prev_submap.id) + "_to_" + std::to_string(frame.id) + "_loop_closure.png";
    cv::imwrite(filename, loop_closure_image);
    std::cout << "Saved loop closure image: " << filename << std::endl;
}

}  // namespace mapper3d