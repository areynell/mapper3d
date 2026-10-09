#pragma once

#include <kiss_matcher/FasterPFH.hpp>
#include <kiss_matcher/KISSMatcher.hpp>
#include <optional>
#include <small_gicp/pcl/pcl_point.hpp>
#include <small_gicp/pcl/pcl_point_traits.hpp>
#include <small_gicp/pcl/pcl_registration.hpp>
#include <sophus/se3.hpp>
#include <unordered_map>

#include "Scancontext.h"
#include "mapper3d/common.h"
#include "mapper3d/utils.h"

namespace mapper3d {

struct LoopClosureDetectorParams {
    // IMPORTANT: This should be the same as the voxel size used in Mapper3D (for both PCL-based downsampling and
    // KISS-Matcher)
    double voxel_size = 0.5;  // Voxel size for KISS-Matcher (m)

    double position_uncertainty = 2.5;                      // Position uncertainty for loop closure detection (m)
    double orientation_uncertainty = utils::deg2rad(40.0);  // Orientation uncertainty for loop closure detection (rad)

    // GICP (small_gicp) refines KISS-Matcher's answer. A point is used in the alignment process only when its nearest
    // point in the other scan is closer than max_correspondence_distance.
    double max_correspondence_distance = 1.0;  // About twice voxel_size (m)
    int num_threads = 4;                       // Threads for GICP (small_gicp's default)

    // A loop closure candidate is rejected unless it passes both.
    size_t min_final_inliers = 8;    // KISS-Matcher's matched keypoints after its final step
    double min_inlier_ratio = 0.70;  // GICP's num_inliers / points in the scan

    double x_loop_uncertainty = 0.05;                     // X position uncertainty for loop closures (m)
    double y_loop_uncertainty = 0.05;                     // Y position uncertainty for loop closures (m)
    double z_loop_uncertainty = 0.05;                     // Z position uncertainty for loop closures (m)
    double roll_loop_uncertainty = utils::deg2rad(3.0);   // Roll orientation uncertainty for loop closures (rad)
    double pitch_loop_uncertainty = utils::deg2rad(3.0);  // Pitch orientation uncertainty for loop closures (rad)
    double yaw_loop_uncertainty = utils::deg2rad(3.0);    // Yaw orientation uncertainty for loop closures (rad)
};

struct LoopClosure {
    size_t prev_keyframe_id;
    size_t keyframe_id;
    Sophus::SE3d relative_pose;
    Eigen::Matrix<double, 6, 6> covariance;

    LoopClosure() = default;

    LoopClosure(size_t prev_keyframe_id, size_t keyframe_id, const Sophus::SE3d& relative_pose,
                const Eigen::Matrix<double, 6, 6>& covariance)
        : prev_keyframe_id(prev_keyframe_id),
          keyframe_id(keyframe_id),
          relative_pose(relative_pose),
          covariance(covariance) {}
};

class LoopClosureDetector {
   public:
    LoopClosureDetector() = delete;
    LoopClosureDetector(const LoopClosureDetectorParams& params);
    ~LoopClosureDetector() = default;

    std::optional<LoopClosure> detect_loop_closures(const Frame& frame);

   private:
    small_gicp::RegistrationResult local_registration(const pcl::PointCloud<pcl::PointXYZ>::Ptr prev_point_cloud,
                                                      const pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud,
                                                      const Sophus::SE3d& initial_transform_guess);

    void update_submaps(const Frame& frame);

    void visualize_point_cloud_alignment(const Submap& prev_submap, const Frame& frame,
                                         const Sophus::SE3d& relative_pose, const size_t image_width,
                                         const size_t image_height, const float view_range_x, const float view_range_y);

    Eigen::Matrix<double, 6, 6> loop_closure_covariance_;

    kiss_matcher::KISSMatcherConfig global_registration_config_;
    kiss_matcher::KISSMatcher global_registration_;
    small_gicp::RegistrationPCL<pcl::PointXYZ, pcl::PointXYZ> local_registration_;

    size_t min_final_inliers_;
    double min_inlier_ratio_;

    Submap current_submap_;  // Current submap being built
    std::unordered_map<size_t, Submap> submaps_;
    std::unordered_map<size_t, Frame> frames_;

    SCManager scan_context_manager_;  // ScanContext manager for loop closure detection
};
}  // namespace mapper3d