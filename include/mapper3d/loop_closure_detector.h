#include <pcl/visualization/pcl_visualizer.h>

#include <kiss_matcher/FasterPFH.hpp>
#include <kiss_matcher/KISSMatcher.hpp>
#include <optional>
#include <sophus/se3.hpp>

#include "mapper3d/common.h"

namespace mapper3d {

struct LoopClosureDetectorParams {
    double distance_threshold = 10.0;   // Distance threshold for loop closure detection (m)
    uint num_inliers_threshold = 10;    // Number of inliers threshold for KISS-Matcher
    uint num_prev_frames_to_skip = 40;  // Number of previous frames to skip for loop closure detection

    double kiss_resolution = 0.2;  // Voxel size for KISS-Matcher (m)
};

struct LoopClosure {
    size_t prev_id;
    size_t id;
    Sophus::SE3d relative_pose;
    Eigen::Matrix<double, 6, 6> covariance;

    LoopClosure(size_t prev_id, size_t id, const Sophus::SE3d& relative_pose,
                const Eigen::Matrix<double, 6, 6>& covariance)
        : prev_id(prev_id), id(id), relative_pose(relative_pose), covariance(covariance) {}
};

class LoopClosureDetector {
   public:
    LoopClosureDetector() = delete;
    LoopClosureDetector(const LoopClosureDetectorParams& params);
    ~LoopClosureDetector() = default;

    std::optional<LoopClosure> find_loop_closures(const Frame& frame);

   private:
    void visualize_point_cloud_alignment(const Frame& prev_frame, const Frame& frame, const Sophus::SE3d& relative_pose,
                                         const size_t image_width, const size_t image_height, const float view_range_x,
                                         const float view_range_y);

    double distance_threshold_;     // Distance threshold for loop closure detection (m)
    uint num_inliers_threshold_;    // Number of inliers threshold for KISS-Matcher
    uint num_prev_frames_to_skip_;  // Number of previous frames to skip for loop closure detection

    double kiss_resolution_;
    kiss_matcher::KISSMatcherConfig kiss_config_;
    kiss_matcher::KISSMatcher kiss_matcher_;

    std::unordered_map<size_t, Frame> frames_;
    std::shared_ptr<pcl::visualization::PCLVisualizer> viewer_;
};
}  // namespace mapper3d