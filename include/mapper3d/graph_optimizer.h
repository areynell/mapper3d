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
#include <thread>

#include "mapper3d/common.h"
#include "mapper3d/loop_closure_detector.h"

namespace mapper3d {

struct GraphOptimizerParams {
    int max_iterations = 100;
    // gtsam::GncLossType loss_type = gtsam::GncLossType::TLS;
    double map_voxel_size = 0.5;  // Voxel resolution of the accumulated map (m)

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
    void add_odometry_constraint(const Sophus::SE3d& pose, const Eigen::Matrix<double, 6, 6>& odometry_covariance);

    void add_loop_closure_constraint(size_t prev_pose_id, size_t pose_id, const Sophus::SE3d& relative_pose,
                                     const Eigen::Matrix<double, 6, 6>& loop_closure_covariance);

    void optimize();

    void save_map_and_graph();

    void generate_optimized_vdb_map();

    void update_map_to_odom_transform();

    void update_frame_poses();

    void update_vdb_map(const pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud, const Sophus::SE3d& sensor_pose);

    // Rebuilds map_snapshot_ from map_vdb_ if the grid changed since the last call.
    // IMPORTANT: The caller must hold map_mutex_.
    pcl::PointCloud<pcl::PointXYZ>::ConstPtr rebuild_map_snapshot();

    std::vector<Sophus::SE3d> extract_online_poses();

    std::vector<std::pair<size_t, size_t>> extract_loop_closure_edges();

    gtsam::LevenbergMarquardtParams lm_params_;
    gtsam::GncParams<gtsam::LevenbergMarquardtParams> gnc_params_;

    size_t current_pose_index_;

    std::mutex graph_mutex_;
    gtsam::NonlinearFactorGraph graph_;
    gtsam::Values unoptimized_estimates_;
    gtsam::Values optimized_estimates_;
    gtsam::Values online_estimates_;
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

    std::thread frame_processing_thread_;

    // // Visualization and debugging
    // std::vector<Sophus::SE3d> online_poses_;
    // std::vector<std::pair<size_t, size_t>> loop_closure_edges_;
};

}  // namespace mapper3d