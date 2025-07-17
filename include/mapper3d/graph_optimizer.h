#pragma once

#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/nonlinear/GncParams.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
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

    void generate_optimized_vdb_map();

    void generate_optimized_point_cloud_map();

    gtsam::Values get_unoptimized_poses();

    gtsam::Values get_optimized_poses();

    gtsam::Values get_online_poses();

    gtsam::NonlinearFactorGraph get_graph();

    pcl::PointCloud<pcl::PointXYZ> get_map();

    Sophus::SE3d map_to_odom_transform();

   private:
    void update_map_to_odom_transform();

    void update_frame_poses();

    void update_vdb_map(const pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud, const Sophus::SE3d& sensor_pose);

    void update_point_cloud_map(const pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud, const Sophus::SE3d& sensor_pose);

    gtsam::LevenbergMarquardtParams lm_params_;
    gtsam::GncParams<gtsam::LevenbergMarquardtParams> gnc_params_;

    size_t current_pose_index_;

    std::mutex graph_mutex_;
    gtsam::NonlinearFactorGraph graph_;
    gtsam::Values initial_estimates_;
    gtsam::Values optimized_estimates_;
    gtsam::Values online_estimates_;
    bool incremental_optimization_;

    bool stop_processing_frames_;
    std::mutex unprocessed_frames_mutex_;
    std::queue<Frame> unprocessed_frames_;
    std::mutex processed_frames_mutex_;
    std::vector<Frame> processed_frames_;
    std::condition_variable frame_processing_cv_;
    std::thread frame_processing_thread_;
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
    pcl::PointCloud<pcl::PointXYZ>::Ptr map_pcd_;
    openvdb::FloatGrid::Ptr map_vdb_;
    float map_voxel_size_;
};

}  // namespace mapper3d