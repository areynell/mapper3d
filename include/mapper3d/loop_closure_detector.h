#include <kiss_matcher/FasterPFH.hpp>
#include <kiss_matcher/KISSMatcher.hpp>
#include <optional>
#include <sophus/se3.hpp>
#include <unordered_map>

#include "Scancontext.h"
#include "mapper3d/common.h"
#include "mapper3d/utils.h"

namespace mapper3d {

struct LoopClosureDetectorParams {
    // double distance_threshold = 10.0;   // Distance threshold for loop closure detection (m)
    double distance_threshold = 20.0;   // Distance threshold for loop closure detection (m)
    uint num_inliers_threshold = 10;    // Number of inliers threshold for KISS-Matcher
    uint num_prev_frames_to_skip = 40;  // Number of previous frames to skip for loop closure detection

    // IMPORTANT: This should be the same as the voxel size used in Mapper3D (for both PCL-based downsampling and
    // KISS-Matcher)
    double voxel_size = 0.5;  // Voxel size for KISS-Matcher (m)
    // double normal_radius = 3.0 * voxel_size;
    // double fpfh_radius = 5.0 * voxel_size;

    double position_uncertainty = 2.5;                      // Position uncertainty for loop closure detection (m)
    double orientation_uncertainty = utils::deg2rad(40.0);  // Orientation uncertainty for loop closure detection (rad)

    double x_loop_uncertainty = 0.05;                     // X position uncertainty for loop closures (m)
    double y_loop_uncertainty = 0.05;                     // Y position uncertainty for loop closures (m)
    double z_loop_uncertainty = 0.05;                     // Z position uncertainty for loop closures (m)
    double roll_loop_uncertainty = utils::deg2rad(3.0);   // Roll orientation uncertainty for loop closures (rad)
    double pitch_loop_uncertainty = utils::deg2rad(3.0);  // Pitch orientation uncertainty for loop closures (rad)
    double yaw_loop_uncertainty = utils::deg2rad(3.0);    // Yaw orientation uncertainty for loop closures (rad)
};

struct LoopClosure {
    size_t prev_id;
    size_t id;
    Sophus::SE3d relative_pose;
    Eigen::Matrix<double, 6, 6> covariance;

    LoopClosure() = default;

    LoopClosure(size_t prev_id, size_t id, const Sophus::SE3d& relative_pose,
                const Eigen::Matrix<double, 6, 6>& covariance)
        : prev_id(prev_id), id(id), relative_pose(relative_pose), covariance(covariance) {}
};

class LoopClosureDetector {
   public:
    LoopClosureDetector() = delete;
    LoopClosureDetector(const LoopClosureDetectorParams& params);
    ~LoopClosureDetector() = default;

    std::optional<LoopClosure> detect_loop_closures(const Frame& frame);

   private:
    void update_submaps(const Frame& frame);

    void visualize_point_cloud_alignment(const Submap& prev_submap, const Frame& frame,
                                         const Sophus::SE3d& relative_pose, const size_t image_width,
                                         const size_t image_height, const float view_range_x, const float view_range_y);

    double distance_threshold_;     // Distance threshold for loop closure detection (m)
    uint num_inliers_threshold_;    // Number of inliers threshold for KISS-Matcher
    uint num_prev_frames_to_skip_;  // Number of previous frames to skip for loop closure detection

    Eigen::Matrix<double, 6, 6> loop_closure_covariance_;

    kiss_matcher::KISSMatcherConfig kiss_config_;
    kiss_matcher::KISSMatcher kiss_matcher_;

    Submap current_submap_;  // Current submap being built
    std::unordered_map<size_t, Submap> submaps_;
    std::unordered_map<size_t, Frame> frames_;  // Used with ScanContext++ for loop closure detection

    SCManager scan_context_manager_;  // ScanContext manager for loop closure detection
};
}  // namespace mapper3d