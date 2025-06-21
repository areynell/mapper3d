#pragma once

#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/nonlinear/GncParams.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>

#include <chrono>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <sophus/se3.hpp>
#include <thread>

#include "mapper3d/common.h"
#include "mapper3d/loop_closure_detector.h"

namespace mapper3d {

struct GraphOptimizerParams {
    int max_iterations = 100;
    // gtsam::GncLossType loss_type = gtsam::GncLossType::TLS;

    LoopClosureDetectorParams loop_closure_detector_params;
};

class GraphOptimizer {
   public:
    GraphOptimizer() = delete;
    GraphOptimizer(const GraphOptimizerParams& params);

    ~GraphOptimizer() = default;

    GraphOptimizer(const GraphOptimizer&) = delete;
    GraphOptimizer& operator=(const GraphOptimizer&) = delete;

    GraphOptimizer(GraphOptimizer&&) = default;
    GraphOptimizer& operator=(GraphOptimizer&&) = default;

    void add_frame(const Frame& frame);

    void process_frames();

    void add_odometry_constraint(const Sophus::SE3d& pose, const Eigen::Matrix<double, 6, 6>& odometry_covariance);

    void add_loop_closure_constraint(size_t prev_pose_id, size_t pose_id, const Sophus::SE3d& relative_pose,
                                     const Eigen::Matrix<double, 6, 6>& loop_closure_covariance);

    void optimize();

    void update_map_to_odom_transform();

    const gtsam::Values& unoptimized_pose_graph() const;

    const gtsam::Values& optimized_pose_graph() const;

    Sophus::SE3d map_to_odom_transform();

   private:
    gtsam::LevenbergMarquardtParams lm_params_;
    gtsam::GncParams<gtsam::LevenbergMarquardtParams> gnc_params_;

    size_t current_pose_index_;

    std::mutex graph_mutex_;
    gtsam::NonlinearFactorGraph graph_;
    gtsam::Values initial_estimates_;
    gtsam::Values optimized_estimates_;
    bool incremental_optimization_;

    bool stop_processing_frames_;
    std::mutex unprocessed_frames_mutex_;
    std::queue<Frame> unprocessed_frames_;
    std::condition_variable frame_processing_cv_;
    std::thread frame_processing_thread_;

    std::chrono::steady_clock::time_point most_recent_optimization_time_;
    std::chrono::duration<double> optimization_trigger_interval_;
    size_t min_new_loop_closures_for_optimization_;
    size_t num_new_loop_closures_;
    LoopClosureDetector loop_closure_detector_;

    std::mutex map_to_odom_transform_mutex_;
    Sophus::SE3d map_to_odom_transform_;
};

}  // namespace mapper3d