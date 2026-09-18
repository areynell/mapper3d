#pragma once

#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/nonlinear/GncParams.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/dataset.h>
#include <openvdb/openvdb.h>
#include <pcl/point_cloud.h>

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <mutex>
#include <queue>
#include <sophus/se3.hpp>
#include <string>
#include <thread>
#include <vector>

#include "mapper3d/common.h"
#include "mapper3d/loop_closure_detector.h"

namespace mapper3d {

struct GraphOptimizerParams {
    int max_iterations = 100;
    // gtsam::GncLossType loss_type = gtsam::GncLossType::TLS;
    double map_voxel_size = 0.5;  // Voxel resolution of the accumulated map (m)

    // Keyframe trajectories written to disk in the TUM format at shutdown for offline evaluation with evo.
    std::string unoptimized_trajectory_file = "unoptimized_keyframes_tum.txt";
    std::string optimized_trajectory_file = "optimized_keyframes_tum.txt";

    LoopClosureDetectorParams loop_closure_detector_params;
};

class GraphOptimizer {
   public:
    GraphOptimizer() = delete;
    GraphOptimizer(const GraphOptimizerParams& params);

    ~GraphOptimizer();

    GraphOptimizer(const GraphOptimizer&) = delete;
    GraphOptimizer& operator=(const GraphOptimizer&) = delete;

    GraphOptimizer(GraphOptimizer&&) = delete;
    GraphOptimizer& operator=(GraphOptimizer&&) = delete;

    void add_frame(const Frame& frame);

    void process_frames();

    gtsam::Values get_unoptimized_poses();

    gtsam::Values get_optimized_poses();

    void get_visualization_data(VisualizationData& data);

    // std::vector<Sophus::SE3d> get_online_poses();

    // std::vector<std::pair<size_t, size_t>> get_loop_closure_edges();

    gtsam::NonlinearFactorGraph get_graph();

    pcl::PointCloud<pcl::PointXYZ>::ConstPtr get_map();

    // std::vector<Eigen::Matrix<double, 6, 6>> get_pose_covariances();

    Sophus::SE3d map_to_odom_transform();

   private:
    void add_odometry_constraint(const Sophus::SE3d& pose, const double timestamp,
                                 const Eigen::Matrix<double, 6, 6>& odometry_covariance);

    void add_loop_closure_constraint(size_t prev_keyframe_id, size_t keyframe_id, const Sophus::SE3d& relative_pose,
                                     const Eigen::Matrix<double, 6, 6>& loop_closure_covariance);

    void optimize();

    void save_results();

    void save_trajectories();

    void write_keyframe_trajectory(const gtsam::Values& estimates, const std::string& path,
                                   const std::string& description);

    // Used for debugging and evaluation against possible ground-truth datasets.
    void write_keyframe_stamps() const;

    void generate_optimized_vdb_map();

    void update_map_to_odom_transform();

    void update_frame_poses();

    void update_vdb_map(const pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud, const Sophus::SE3d& sensor_pose);

    pcl::PointCloud<pcl::PointXYZ>::ConstPtr rebuild_map_snapshot();

    std::vector<Sophus::SE3d> extract_online_poses();

    std::vector<std::pair<size_t, size_t>> extract_loop_closure_edges();

    gtsam::LevenbergMarquardtParams lm_params_;
    gtsam::GncParams<gtsam::LevenbergMarquardtParams> gnc_params_;

    // Must stay in lockstep with Mapper3D::keyframe_id_, because Frame::keyframe_id is looked up as
    // gtsam::Symbol('x', id) in update_frame_poses() and in every loop-closure edge.
    size_t current_keyframe_index_;

    std::mutex graph_mutex_;
    gtsam::NonlinearFactorGraph graph_;
    // The graph optimizer's initial guess provided by lidar odometry poses in the odom frame.
    gtsam::Values unoptimized_estimates_;
    // The result of the most recent graph optimization in the map frame.
    gtsam::Values optimized_estimates_;
    // Same result as optimized_estimates_, but updated incrementally with new lidar odometry
    // poses transformed into the map frame, to provide a more up-to-date estimate for visualization.
    gtsam::Values online_estimates_;
    // Scan timestamp of each keyframe, which makes the saved trajectories matchable to ground truth.
    std::vector<double> keyframe_timestamps_;
    bool incremental_optimization_;

    bool stop_processing_frames_;
    std::mutex unprocessed_frames_mutex_;
    std::queue<Frame> unprocessed_frames_;
    std::mutex processed_frames_mutex_;
    std::vector<Frame> processed_frames_;
    std::condition_variable frame_processing_cv_;
    std::chrono::steady_clock::time_point latest_frame_received_time_;

    std::chrono::steady_clock::time_point latest_optimization_time_;
    std::chrono::duration<double> optimization_trigger_interval_;
    size_t min_new_loop_closures_for_optimization_;
    size_t num_new_loop_closures_;
    size_t num_loop_closures_;
    LoopClosureDetector loop_closure_detector_;

    std::mutex map_to_odom_transform_mutex_;
    Sophus::SE3d map_to_odom_transform_;

    std::mutex map_mutex_;
    openvdb::BoolGrid::Ptr map_vdb_;
    double map_voxel_size_;

    pcl::PointCloud<pcl::PointXYZ>::ConstPtr map_snapshot_;
    bool map_changed_;

    std::string unoptimized_trajectory_file_;
    std::string optimized_trajectory_file_;

    std::thread frame_processing_thread_;

    // // Visualization and debugging
    // std::vector<Sophus::SE3d> online_poses_;
    // std::vector<std::pair<size_t, size_t>> loop_closure_edges_;
};

}  // namespace mapper3d