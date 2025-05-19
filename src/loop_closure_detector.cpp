#include "mapper3d/loop_closure_detector.h"

#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "mapper3d/common.h"
#include "mapper3d/utils.h"

namespace mapper3d {

LoopClosureDetector::LoopClosureDetector(const LoopClosureDetectorParams& params)
    : kiss_resolution_(params.kiss_resolution),
      kiss_config_(kiss_resolution_),
      kiss_matcher_(kiss_config_),
      distance_threshold_(params.distance_threshold),
      num_inliers_threshold_(params.num_inliers_threshold),
      num_prev_frames_to_skip_(params.num_prev_frames_to_skip) {}

std::optional<LoopClosure> LoopClosureDetector::find_loop_closures(const Frame& frame) {
    std::cout << "Searching for loop closures..." << std::endl;

    // TODO: Optimize this by first computing FasterPFH features and storing them in the frame
    frames_.emplace(frame.id, frame);

    std::cout << "frames_.size() = " << frames_.size() << std::endl;

    // clang-format off
    // Loop closure detection logic is as follows:
    // 1) Compare the current frame with all previous frames
    // 2) If the current frame has a position difference < threshold with a previous frame, then try and match the two frames using KISS-Matcher global registration
    // 3) If the two frames match within a certain threshold, then perform a GICP registration to refine the match
    // 4) If the GICP registration is successful, add the loop closure to the graph optimizer
    // 5) Return the loop closure
    // clang-format on

    for (const auto& [prev_id, prev_frame] : frames_) {
        if (frame.id == prev_id) {
            continue;  // Skip the same frame
        }
        // Check that current frame isn't matched with a previous frame that is too close in time
        if ((frame.id - prev_id) < num_prev_frames_to_skip_) {
            continue;  // Skip frames that are too close
        }

        // Check if the distance between the two frames is below a certain threshold
        const double distance = (frame.pose.translation() - prev_frame.pose.translation()).norm();
        if (distance < distance_threshold_) {
            const auto& prev_point_cloud_vec = utils::point_cloud_to_vec(prev_frame.point_cloud);
            const auto& point_cloud_vec = utils::point_cloud_to_vec(frame.point_cloud);
            if (!prev_point_cloud_vec.empty() && !point_cloud_vec.empty()) {
                // Perform KISS-Matcher global registration
                const auto solution = kiss_matcher_.estimate(point_cloud_vec, prev_point_cloud_vec);

                std::cout << "kiss_matcher_.getNumFinalInliers() = " << kiss_matcher_.getNumFinalInliers()
                          << ", num_inliers_threshold_ = " << num_inliers_threshold_ << std::endl;

                if (kiss_matcher_.getNumFinalInliers() > num_inliers_threshold_) {
                    // // Perform GICP local registration to refine the match
                    // const Sophus::SE3d refined_pose_change =
                    //     small_gicp_registration(point_cloud_vec, prev_point_cloud_vec, solution.relative_pose);
                    // if (refined_pose_change.is_valid()) {
                    //     return LoopClosure(prev_id, frame.id, refined_pose_change, Eigen::Matrix<double, 6,
                    //     6>::Identity());
                    // }

                    const Sophus::SE3d relative_pose = Sophus::SE3d(solution.rotation, solution.translation);

                    visualize_point_cloud_alignment(prev_frame, frame, relative_pose, 1920, 1080, 200.0F, 200.0F);

                    return LoopClosure(prev_id, frame.id, relative_pose, Eigen::Matrix<double, 6, 6>::Identity());

                    // TODO: Return the best of N matches instead of the first valid match
                }
            }
        }
    }

    std::cout << "\033[31m" << "No loop closures found for current frame " << "\033[0m" << std::endl;

    return std::nullopt;  // No loop closure found
}

void LoopClosureDetector::visualize_point_cloud_alignment(const Frame& prev_frame, const Frame& frame,
                                                          const Sophus::SE3d& relative_pose, const size_t image_width,
                                                          const size_t image_height, const float view_range_x,
                                                          const float view_range_y) {
    // Transform previous point cloud from sensor frame to world frame
    pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_prev_point_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::transformPointCloud(*prev_frame.point_cloud, *transformed_prev_point_cloud,
                             prev_frame.pose.matrix().cast<double>());

    // Transform current point cloud from sensor frame to world frame
    pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_point_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    const Sophus::SE3d corrected_sensor_pose = prev_frame.pose * relative_pose;
    pcl::transformPointCloud(*frame.point_cloud, *transformed_point_cloud,
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
    std::string filename = std::to_string(prev_frame.id) + "_to_" + std::to_string(frame.id) + "_loop_closure.png";
    cv::imwrite(filename, loop_closure_image);
    std::cout << "Saved loop closure image: " << filename << std::endl;
}

}  // namespace mapper3d