#pragma once

#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

// include opencv
#include <Eigen/Dense>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <sophus/se3.hpp>
#include <unordered_map>

#include "mapper3d/logging.h"
#include "mapper3d/utils.h"

namespace mapper3d {

// constexpr double kDefaultVoxelSize = 0.25;  // Default voxel size for downsampling (m)

struct Frame {
    size_t id;
    Sophus::SE3d pose;
    std::vector<Eigen::Vector3f> point_cloud;
    Eigen::Matrix<double, 6, 6> odometry_covariance;

    Frame() {}

    Frame(size_t id, const Sophus::SE3d &pose, const std::vector<Eigen::Vector3f> &point_cloud,
          const Eigen::Matrix<double, 6, 6> &odometry_covariance)
        : id(id), pose(pose), point_cloud(point_cloud), odometry_covariance(odometry_covariance) {}
};

struct Submap {
    size_t id;
    size_t num_frames = 0;  // Number of frames in the submap
    // TODO: Think about using different ways to decide whether to finalize a submap (e.g. distance traveled, time
    // elapsed, etc.)
    size_t max_num_frames = 25;  // Default number of frames in a submap
    // IMPORTANT: This should be the same as the voxel size used in Mapper3D and LoopClosureDetector
    double voxel_size = 0.5;    // Voxel size for downsampling (m)
    bool is_finalized = false;  // Whether the submap is finalized

    Sophus::SE3d origin_pose;
    pcl::PointCloud<pcl::PointXYZ>::Ptr accumulated_point_cloud;
    pcl::PointCloud<pcl::PointXYZ>::Ptr downsampled_point_cloud;
    std::vector<Eigen::Vector3f> downsampled_point_cloud_vec;  // Only need this because of KISS-Matcher and small_gicp

    // HYPOTHESIS: If we add loop closures from point cloud poses to the closest submap pose, we should get better
    // results than matching point cloud poses to the origin pose of the submap.
    std::unordered_map<size_t, Sophus::SE3d> poses;  // Maps frame ID to its pose in the submap

    Submap() : id(0), origin_pose(Sophus::SE3d()), accumulated_point_cloud(new pcl::PointCloud<pcl::PointXYZ>()) {}

    Submap(const size_t id, const Sophus::SE3d &origin_pose, const pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud)
        : id(id), origin_pose(origin_pose), accumulated_point_cloud(new pcl::PointCloud<pcl::PointXYZ>()) {
        *accumulated_point_cloud = *point_cloud;  // Deep copy
        downsampled_point_cloud = downsample_point_cloud(accumulated_point_cloud, voxel_size);
        num_frames = 1;
    }

    Submap(const size_t id, const size_t max_num_frames, const double voxel_size, const Sophus::SE3d &origin_pose,
           const pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud)
        : id(id),
          max_num_frames(max_num_frames),
          voxel_size(voxel_size),
          origin_pose(origin_pose),
          accumulated_point_cloud(new pcl::PointCloud<pcl::PointXYZ>()) {
        *accumulated_point_cloud = *point_cloud;  // Deep copy
        downsampled_point_cloud = downsample_point_cloud(accumulated_point_cloud, voxel_size);
        num_frames = 1;
    }

    void add_point_cloud(const size_t id, const Sophus::SE3d &sensor_pose,
                         const pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud) {
        Timer timer(__func__, LogLevel::Debug);

        // if (!is_initialized) {
        if (num_frames == 0) {
            std::cerr << "Submap has not been initialized with a valid ID. Cannot add point cloud." << std::endl;
            return;
        }
        if (num_frames >= max_num_frames && !is_finalized) {
            finalize_submap();
        }

        // Transform the point cloud to the submap's origin frame
        pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZ>());
        const Sophus::SE3d relative_pose = origin_pose.inverse() * sensor_pose;
        pcl::transformPointCloud(*point_cloud, *transformed_cloud, relative_pose.matrix());

        // Add the transformed point cloud to the accumulated point cloud
        *accumulated_point_cloud += *transformed_cloud;

        // Store the point cloud's pose in the submap, so we can find the closest pose later during loop closure
        // detection
        poses[id] = sensor_pose;

        num_frames++;
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr downsample_point_cloud(pcl::PointCloud<pcl::PointXYZ>::Ptr &point_cloud,
                                                               const float voxel_size) {
        Timer timer(__func__, LogLevel::Debug);

        pcl::VoxelGrid<pcl::PointXYZ> voxel_grid;
        voxel_grid.setLeafSize(voxel_size, voxel_size, voxel_size);
        voxel_grid.setInputCloud(point_cloud);

        pcl::PointCloud<pcl::PointXYZ>::Ptr downsampled_cloud(new pcl::PointCloud<pcl::PointXYZ>());
        voxel_grid.filter(*downsampled_cloud);

        return downsampled_cloud;
    }

    void finalize_submap() {
        Timer timer(__func__, LogLevel::Debug);

        downsampled_point_cloud = downsample_point_cloud(accumulated_point_cloud, voxel_size);
        downsampled_point_cloud_vec = utils::point_cloud_to_vec(downsampled_point_cloud);
        // Clear the accumulated point cloud if the submap is finalized, since we no longer need it
        accumulated_point_cloud->clear();

        is_finalized = true;

        visualize_submap(1920, 1080, 400.0F, 400.0F);  // Visualize the submap after finalization

        assert(downsampled_point_cloud_vec.size() == downsampled_point_cloud->size());
    }

    std::pair<size_t, Sophus::SE3d> find_closest_pose(const Sophus::SE3d &pose) const {
        const Eigen::Vector3d query_translation = pose.translation();

        // Find the closest pose in the submap to the current frame
        Sophus::SE3d closest_pose = origin_pose;
        size_t closest_pose_id = id;
        double min_distance_squared = (query_translation - origin_pose.translation()).squaredNorm();

        // Check all poses in the submap to find the closest one
        for (const auto &[prev_id, prev_pose] : poses) {
            const double distance_squared = (query_translation - prev_pose.translation()).squaredNorm();
            if (distance_squared < min_distance_squared) {
                min_distance_squared = distance_squared;
                closest_pose = prev_pose;
                closest_pose_id = prev_id;
            }
        }

        return std::make_pair(closest_pose_id, closest_pose);
    }

    bool initialized() const {
        if (num_frames == 0) {
            return false;
        }
        return true;
    }

    bool finalized() const { return is_finalized; }

    void visualize_submap(const size_t image_width, const size_t image_height, const float view_range_x,
                          const float view_range_y) {
        // Transform previous point cloud from sensor frame to world frame
        const pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud = utils::point_vec_to_cloud(downsampled_point_cloud_vec);
        pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_point_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::transformPointCloud(*point_cloud, *transformed_point_cloud, origin_pose.matrix().cast<double>());

        // Get the current submap's position to center the view
        const Eigen::Vector3d current_submap_position = origin_pose.translation();
        const float center_x = static_cast<float>(current_submap_position.x());
        const float center_y = static_cast<float>(current_submap_position.y());

        // Calculate world coordinate bounds centered on the current submap's pose
        const float x_min = center_x - view_range_x / 2.0f;
        const float x_max = center_x + view_range_x / 2.0f;
        const float y_min = center_y - view_range_y / 2.0f;
        const float y_max = center_y + view_range_y / 2.0f;

        // Ensure (x_max - x_min) and (y_max - y_min) are not zero to avoid division by zero
        if (x_max <= x_min || y_max <= y_min) {
            std::cerr << "Error: Invalid world bounds for visualization (x_max <= x_min or y_max <= y_min)."
                      << std::endl;
            return;
        }
        // Calculate scaling factors based on the chosen fixed bounds
        const float x_scale = static_cast<float>(image_width) / (x_max - x_min);
        const float y_scale = static_cast<float>(image_height) / (y_max - y_min);

        // Function to map a 3D point (using X and Y for top-down) to a 2D pixel coordinate
        auto map_point_to_pixel = [&](const pcl::PointXYZ &point) -> cv::Point {
            const int px = static_cast<int>((point.x - x_min) * x_scale);
            // Flip y-axis (image coordinates have origin at top-left, world Y might be up or forward)
            const int py = image_height - 1 - static_cast<int>((point.y - y_min) * y_scale);
            return cv::Point(px, py);
        };

        cv::Mat submap_image(image_height, image_width, CV_8UC3, cv::Scalar(0, 0, 0));

        // Draw current point cloud
        for (const auto &point : transformed_point_cloud->points) {
            cv::Point pixel = map_point_to_pixel(point);
            if (pixel.x >= 0 && pixel.x < image_width && pixel.y >= 0 && pixel.y < image_height) {
                cv::circle(submap_image, pixel, 2, cv::Scalar(0, 255, 0), -1);
            }
        }

        // const std::string window_name = "Top-Down Aligned Point Clouds";
        // cv::imshow(window_name, image);
        // cv::waitKey(0);                  // Wait for a key press to close the window
        // cv::destroyWindow(window_name);  // Clean up the window

        // Save image to disk
        std::string filename = "submap_" + std::to_string(id) + ".png";
        cv::imwrite(filename, submap_image);
        std::cout << "Saved submap image: " << filename << std::endl;
    }
};
}  // namespace mapper3d