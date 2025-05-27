#include "mapper3d/graph_optimizer.h"

#include <gtsam/nonlinear/GncOptimizer.h>
#include <gtsam/nonlinear/GncParams.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/dataset.h>

#include <mutex>

#include "mapper3d/common.h"

namespace mapper3d {

GraphOptimizer::GraphOptimizer(const GraphOptimizerParams& params)
    : loop_closure_detector_(params.loop_closure_detector_params) {
    lm_params_.setMaxIterations(params.max_iterations);

    gnc_params_ = gtsam::GncParams<gtsam::LevenbergMarquardtParams>(lm_params_);
    // gnc_params_.setLossType(params.loss_type);
    gnc_params_.setLossType(gtsam::GncLossType::TLS);

    current_pose_index_ = 0;
    stop_processing_frames_ = false;

    // Start processing frames in a separate thread
    frame_processing_thread_ = std::thread(&GraphOptimizer::process_frames, this);
}

void GraphOptimizer::add_frame(const Frame& frame) {
    {
        std::scoped_lock<std::mutex> lock(unprocessed_frames_mutex_);
        unprocessed_frames_.push(frame);
    }

    // Notify the processing thread that a new frame is available
    frame_processing_cv_.notify_one();
}

void GraphOptimizer::process_frames() {
    Frame frame;
    bool new_frame_available = false;

    while (!stop_processing_frames_) {
        Timer timer("GraphOptimizer::process_frames");

        new_frame_available = false;
        {
            std::unique_lock<std::mutex> lock(unprocessed_frames_mutex_);
            // Wait until:
            // 1. There's a frame in the queue, OR
            // 2. The stop_processing_frames_ is set.
            frame_processing_cv_.wait(lock, [this] { return !unprocessed_frames_.empty() || stop_processing_frames_; });

            // If stop is signaled and the queue is empty, exit the loop.
            if (stop_processing_frames_ && unprocessed_frames_.empty()) {
                break;
            }

            // Process the frame if available
            if (!unprocessed_frames_.empty()) {
                std::cout << "unprocessed_frames_.size() = " << unprocessed_frames_.size() << std::endl;

                frame = unprocessed_frames_.front();
                unprocessed_frames_.pop();
                new_frame_available = true;
            }
        }

        if (new_frame_available) {
            std::scoped_lock<std::mutex> lock(graph_mutex_);
            add_odometry_constraint(frame.pose, frame.odometry_covariance);

            const auto loop_closure = loop_closure_detector_.detect_loop_closures(frame);
            if (loop_closure) {
                add_loop_closure_constraint(loop_closure->prev_id, loop_closure->id, loop_closure->relative_pose,
                                            loop_closure->covariance);
            }
        }

        std::cout << "Finished processing frame." << std::endl;
    }
}

void GraphOptimizer::add_odometry_constraint(const Sophus::SE3d& pose,
                                             const Eigen::Matrix<double, 6, 6>& odometry_covariance) {
    // std::scoped_lock<std::mutex> lock(graph_mutex_);

    const gtsam::Symbol current_pose_symbol('x', current_pose_index_);

    const gtsam::Pose3 current_pose(gtsam::Rot3(pose.unit_quaternion()), gtsam::Point3(pose.translation()));
    const auto odometry_noise_model = gtsam::noiseModel::Gaussian::Covariance(odometry_covariance);

    if (current_pose_index_ == 0) {
        // Add prior factor for the first pose in the graph
        const Eigen::Matrix<double, 6, 6> prior_covariance =
            Eigen::Matrix<double, 6, 6>::Identity() * 1.0e-9;  // Very small values
        graph_.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(current_pose_symbol, current_pose, prior_covariance);

        // graph_.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(current_pose_symbol, current_pose,
        //                                                         odometry_noise_model);
    } else {
        const gtsam::Symbol prev_pose_symbol('x', current_pose_index_ - 1);

        // Relative pose between previous pose and current pose
        const gtsam::Pose3 prev_pose = initial_estimates_.at<gtsam::Pose3>(prev_pose_symbol);
        const gtsam::Pose3 odometry_relative_pose = prev_pose.inverse() * current_pose;

        // Add odometry factor for subsequent poses
        graph_.emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(prev_pose_symbol, current_pose_symbol,
                                                                  odometry_relative_pose, odometry_noise_model);
    }
    initial_estimates_.insert(current_pose_symbol, current_pose);

    current_pose_index_++;

    // TODO: Remove this once no longer debugging, since it's extremely expensive to write the graph to file
    gtsam::writeG2o(graph_, initial_estimates_, "unoptimized_graph.g2o");
}

void GraphOptimizer::add_loop_closure_constraint(size_t prev_pose_id, size_t pose_id, const Sophus::SE3d& relative_pose,
                                                 const Eigen::Matrix<double, 6, 6>& loop_closure_covariance) {
    // std::scoped_lock<std::mutex> lock(graph_mutex_);

    const gtsam::Symbol prev_pose_symbol('x', prev_pose_id);
    const gtsam::Symbol current_pose_symbol('x', pose_id);

    const gtsam::Pose3 loop_closure_relative_pose(gtsam::Rot3(relative_pose.unit_quaternion()),
                                                  gtsam::Point3(relative_pose.translation()));
    const auto loop_closure_noise_model = gtsam::noiseModel::Gaussian::Covariance(loop_closure_covariance);

    // Add the loop closure factor
    graph_.emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(prev_pose_symbol, current_pose_symbol,
                                                              loop_closure_relative_pose, loop_closure_noise_model);

    // TODO: Remove this once no longer debugging, since it's extremely expensive to write the graph to file
    gtsam::writeG2o(graph_, initial_estimates_, "unoptimized_graph.g2o");
}

void GraphOptimizer::optimize() {
    // std::scoped_lock<std::mutex> lock(graph_mutex_);

    std::cout << "Optimizing pose graph..." << std::endl;

    // gtsam::GncOptimizer<gtsam::GncParams<gtsam::LevenbergMarquardtParams>> optimizer(graph_, initial_estimates_);
    gtsam::GncOptimizer<gtsam::GncParams<gtsam::LevenbergMarquardtParams>> optimizer(graph_, initial_estimates_,
                                                                                     gnc_params_);

    optimized_estimates_ = optimizer.optimize();

    std::cout << "Pose graph optimized." << std::endl;

    // Save the optimized graph to a file
    gtsam::writeG2o(graph_, optimized_estimates_, "optimized_graph.g2o");
}
}  // namespace mapper3d