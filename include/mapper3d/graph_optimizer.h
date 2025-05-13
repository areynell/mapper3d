#pragma once

#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/nonlinear/GncParams.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>

#include <condition_variable>
#include <mutex>
#include <queue>
#include <sophus/se3.hpp>
#include <thread>

#include "mapper3d/common.h"

namespace mapper3d {

struct GraphOptimizerParams {
    int max_iterations = 100;
    // gtsam::GncLossType loss_type = gtsam::GncLossType::TLS;
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

   private:
    gtsam::LevenbergMarquardtParams lm_params_;
    gtsam::GncParams<gtsam::LevenbergMarquardtParams> gnc_params_;

    gtsam::NonlinearFactorGraph graph_;
    gtsam::Values initial_estimates_;
    gtsam::Values optimized_estimates_;
    size_t current_pose_index_;

    std::mutex unprocessed_frames_mutex_;
    std::queue<Frame> unprocessed_frames_;
    std::condition_variable frame_processing_cv_;

    bool stop_processing_frames_;
    std::thread frame_processing_thread_;
    std::mutex graph_mutex_;
};

}  // namespace mapper3d