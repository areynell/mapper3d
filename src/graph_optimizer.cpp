#include "mapper3d/graph_optimizer.h"

#include <gtsam/nonlinear/GncOptimizer.h>
#include <gtsam/nonlinear/GncParams.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/dataset.h>
#include <pcl/io/ply_io.h>

#include <exception>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <string>

#include "mapper3d/logging.h"
#include "mapper3d/utils.h"

namespace mapper3d {

GraphOptimizer::GraphOptimizer(const GraphOptimizerParams& params)
    : loop_closure_detector_(params.loop_closure_detector_params) {
    lm_params_.setMaxIterations(params.max_iterations);
    // lm_params_.setVerbosityLM("VALUES");

    gnc_params_ = gtsam::GncParams<gtsam::LevenbergMarquardtParams>(lm_params_);
    // gnc_params_.setLossType(params.loss_type);

    // GNC TLS struggles on Kitti 19 sequence, where "standard" g2o GM does not. Why?
    // GNC GM struggles on Kitti 13 sequence, where "standard" g2o GM does not. Why?
    gnc_params_.setLossType(gtsam::GncLossType::TLS);
    // gnc_params_.setLossType(gtsam::GncLossType::GM);
    gnc_params_.setVerbosityGNC(gtsam::GncParams<gtsam::LevenbergMarquardtParams>::Verbosity::MU);

    // gtsam::Pose3 orders its tangent space rotation FIRST, then translation.
    gtsam::Vector6 prior_sigmas;
    prior_sigmas << gtsam::Vector3::Constant(params.prior_rotation_sigma),
        gtsam::Vector3::Constant(params.prior_translation_sigma);
    prior_noise_model_ = gtsam::noiseModel::Diagonal::Sigmas(prior_sigmas);

    current_keyframe_index_ = 0;
    stop_processing_frames_ = false;

    // TODO: Make these parameters configurable via GraphOptimizerParams
    latest_frame_received_time_ = std::chrono::steady_clock::now();
    latest_optimization_time_ = std::chrono::steady_clock::now();
    optimization_trigger_interval_ = std::chrono::duration<double>(10.0);  // seconds
    min_new_loop_closures_for_optimization_ = 10;
    num_new_loop_closures_ = 0;
    num_loop_closures_ = 0;

    // TODO: Implement incremental optimization if needed
    incremental_optimization_ = false;

    map_to_odom_transform_ = Sophus::SE3d();

    // Initialize OpenVDB for volumetric map representation
    openvdb::initialize();
    map_voxel_size_ = params.map_voxel_size;
    map_vdb_ = openvdb::BoolGrid::create();
    map_vdb_->setTransform(openvdb::math::Transform::createLinearTransform(map_voxel_size_));
    map_changed_ = false;

    unoptimized_trajectory_file_ = params.unoptimized_trajectory_file;
    optimized_trajectory_file_ = params.optimized_trajectory_file;

    // Start processing frames in a separate thread
    frame_processing_thread_ = std::thread(&GraphOptimizer::process_frames, this);
}

GraphOptimizer::~GraphOptimizer() {
    {
        std::scoped_lock<std::mutex> lock(unprocessed_frames_mutex_);
        stop_processing_frames_ = true;
    }

    // Wake the processing thread. It processes any frames that remain in the queue, writes the
    // final map and graph to disk, then exits its loop.
    frame_processing_cv_.notify_all();

    if (frame_processing_thread_.joinable()) {
        frame_processing_thread_.join();
    }
}

void GraphOptimizer::add_frame(const Frame& frame) {
    {
        std::scoped_lock<std::mutex> lock(unprocessed_frames_mutex_);

        unprocessed_frames_.push(frame);
        latest_frame_received_time_ = std::chrono::steady_clock::now();
    }

    // Notify the processing thread that a new frame is available
    frame_processing_cv_.notify_one();
}

void GraphOptimizer::process_frames() {
    while (true) {
        // IMPORTANT: add_frame() writes unprocessed_frames_ and latest_frame_received_time_ from the ROS callback
        // thread. Everything this loop needs from that shared state is snapshotted into these locals inside the
        // critical section below, so that nothing outside the lock reads it directly.
        Frame frame;
        bool new_frame_available = false;
        bool stop_requested = false;
        bool queue_empty = true;
        std::chrono::steady_clock::time_point latest_frame_received_time;

        {
            std::unique_lock<std::mutex> lock(unprocessed_frames_mutex_);
            // Wait until:
            // 1) There's a frame in the queue, OR
            // 2) The stop_processing_frames_ is set, OR
            // 3) The optimization_trigger_interval has passed since the last optimization
            frame_processing_cv_.wait_for(lock, optimization_trigger_interval_,
                                          [this] { return !unprocessed_frames_.empty() || stop_processing_frames_; });

            stop_requested = stop_processing_frames_;

            // Process the frame if available
            if (!unprocessed_frames_.empty()) {
                Logger::debug("unprocessed_frames_.size() = " + std::to_string(unprocessed_frames_.size()));

                frame = std::move(unprocessed_frames_.front());
                unprocessed_frames_.pop();
                new_frame_available = true;
            }

            queue_empty = unprocessed_frames_.empty();
            latest_frame_received_time = latest_frame_received_time_;
        }

        // NOTE: Declared after the wait so that the timer measures the work, not the idle wait.
        Timer timer(__func__, LogLevel::Debug);

        if (new_frame_available) {
            add_odometry_constraint(frame.pose, frame.timestamp, frame.odometry_covariance, frame.odometry_degraded);

            const auto loop_closure = loop_closure_detector_.detect_loop_closures(frame);
            if (loop_closure) {
                add_loop_closure_constraint(loop_closure->prev_keyframe_id, loop_closure->keyframe_id,
                                            loop_closure->relative_pose, loop_closure->covariance);
            }

            {
                // Incremental map updating
                std::scoped_lock<std::mutex> lock(map_mutex_);

                const auto sensor_pose_odom_frame = frame.pose;
                const auto sensor_pose_map_frame = map_to_odom_transform_ * sensor_pose_odom_frame;
                update_vdb_map(frame.downsampled_point_cloud, sensor_pose_map_frame);
            }

            {
                std::scoped_lock<std::mutex> lock(processed_frames_mutex_);
                processed_frames_.emplace_back(frame);
            }
        }

        const auto current_time = std::chrono::steady_clock::now();
        const auto time_since_latest_optimization = current_time - latest_optimization_time_;
        const auto time_since_latest_frame_received = current_time - latest_frame_received_time;

        Logger::info("Time since latest optimization: " +
                     std::to_string(std::chrono::duration<double>(time_since_latest_optimization).count()) +
                     " seconds.");

        // Perform optimization if there are more than N new loop closures since last optimization
        if ((num_new_loop_closures_ >= min_new_loop_closures_for_optimization_) &&
            (time_since_latest_optimization > optimization_trigger_interval_)) {
            optimize();
        }
        // Perform optimization if there are no unprocessed frames and enough time has passed since the last frame
        else if ((num_new_loop_closures_ > 0) && queue_empty &&
                 (time_since_latest_frame_received > optimization_trigger_interval_)) {
            optimize();
        }

        // Shutdown: no frames remain in the queue. Optimize once more, write the final results to disk,
        // then exit. The check sits after the frame is processed, so that a frame popped in this
        // iteration is never discarded.
        if (stop_requested && queue_empty) {
            // Optimize again if keyframes were added since the last optimization, so save_results() writes
            // every keyframe and not just those up to the last mid-run optimization. A new loop closure is
            // always added with a new keyframe, so this covers new loop closures too.
            const bool keyframes_added_since_optimization = optimized_estimates_.size() < current_keyframe_index_;
            if (num_loop_closures_ > 0 && keyframes_added_since_optimization) {
                optimize();
            }
            save_results();
            break;
        }
    }
}

void GraphOptimizer::add_odometry_constraint(const Sophus::SE3d& pose, const double timestamp,
                                             const Eigen::Matrix<double, 6, 6>& odometry_covariance,
                                             const bool odometry_degraded) {
    std::scoped_lock<std::mutex> lock(graph_mutex_);

    const gtsam::Symbol current_keyframe_symbol('x', current_keyframe_index_);
    keyframe_timestamps_.emplace_back(timestamp);

    // if (incremental_optimization_) {
    //     const Sophus::SE3d pose_in_map_frame = map_to_odom_transform_ * pose;
    //     const gtsam::Pose3 current_pose(gtsam::Rot3(pose_in_map_frame.unit_quaternion()),
    //                                     gtsam::Point3(pose_in_map_frame.translation()));
    // }

    const gtsam::Pose3 current_pose(gtsam::Rot3(pose.unit_quaternion()), gtsam::Point3(pose.translation()));

    if (current_keyframe_index_ == 0) {
        // Add prior factor for the first pose in the graph
        graph_.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(current_keyframe_symbol, current_pose,
                                                                prior_noise_model_);
        // GNC is told which factors to keep by their position in graph_.
        const size_t factor_index = graph_.size() - 1;
        known_inlier_factors_.push_back(factor_index);
    } else {
        const gtsam::Symbol prev_keyframe_symbol('x', current_keyframe_index_ - 1);

        // Relative pose between previous pose and current pose
        const gtsam::Pose3 prev_pose = unoptimized_estimates_.at<gtsam::Pose3>(prev_keyframe_symbol);
        const gtsam::Pose3 odometry_relative_pose = prev_pose.inverse() * current_pose;
        const auto odometry_noise_model = gtsam::noiseModel::Gaussian::Covariance(odometry_covariance);

        // Add odometry factor for subsequent poses
        graph_.emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(prev_keyframe_symbol, current_keyframe_symbol,
                                                                  odometry_relative_pose, odometry_noise_model);

        // GNC must keep a normal odometry edge. A degraded one (registration failed or scans were dropped between its
        // keyframes) may be wrong where a loop closure is right, so GNC is allowed to reject it.
        const size_t factor_index = graph_.size() - 1;
        if (odometry_degraded) {
            degraded_odometry_factors_.push_back(factor_index);
            Logger::warn("Odometry edge " + prev_keyframe_symbol.string() + " -> " + current_keyframe_symbol.string() +
                         " is degraded: registration failed or scans were dropped since the previous keyframe. GNC "
                         "may reject it.");
        } else {
            known_inlier_factors_.push_back(factor_index);
        }
    }
    unoptimized_estimates_.insert(current_keyframe_symbol, current_pose);

    // TODO: Is there a better way to create a pose graph that can be visualized live in RViz as it is being built?
    const Sophus::SE3d online_pose_in_map_frame = map_to_odom_transform_ * pose;
    const gtsam::Pose3 online_pose(gtsam::Rot3(online_pose_in_map_frame.unit_quaternion()),
                                   gtsam::Point3(online_pose_in_map_frame.translation()));
    online_estimates_.insert(current_keyframe_symbol, online_pose);

    // // Used for visualization
    // online_poses_.emplace_back(online_pose_in_map_frame);

    current_keyframe_index_++;
}

void GraphOptimizer::add_loop_closure_constraint(size_t prev_keyframe_id, size_t keyframe_id,
                                                 const Sophus::SE3d& relative_pose,
                                                 const Eigen::Matrix<double, 6, 6>& loop_closure_covariance) {
    std::scoped_lock<std::mutex> lock(graph_mutex_);

    const gtsam::Symbol prev_keyframe_symbol('x', prev_keyframe_id);
    const gtsam::Symbol current_keyframe_symbol('x', keyframe_id);

    const gtsam::Pose3 loop_closure_relative_pose(gtsam::Rot3(relative_pose.unit_quaternion()),
                                                  gtsam::Point3(relative_pose.translation()));
    const auto loop_closure_noise_model = gtsam::noiseModel::Gaussian::Covariance(loop_closure_covariance);

    // Add loop closure factor. GNC may reject it, so it is not a known inlier.
    graph_.emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(prev_keyframe_symbol, current_keyframe_symbol,
                                                              loop_closure_relative_pose, loop_closure_noise_model);
    const size_t factor_index = graph_.size() - 1;
    loop_closure_factors_.push_back(factor_index);

    // // Used for visualization
    // loop_closure_edges_.emplace_back(prev_keyframe_id, keyframe_id);

    num_new_loop_closures_++;
    num_loop_closures_++;
}

void GraphOptimizer::optimize() {
    Timer timer(__func__, LogLevel::InfoImportant);

    Logger::info_important("Optimizing pose graph...");

    // IMPORTANT: Use a copy of gnc_params_. setKnownInliers() appends rather than replaces, so calling it on
    // gnc_params_ would pile up duplicate indices with every optimization.
    gtsam::GncParams<gtsam::LevenbergMarquardtParams> gnc_params = gnc_params_;
    gnc_params.setKnownInliers(known_inlier_factors_);

    // IMPORTANT: The solve runs unlocked, so a long optimization never blocks the reader threads. This is
    // safe because the frame-processing thread is the sole writer of graph_ and the estimates.
    // gtsam::GncOptimizer<gtsam::GncParams<gtsam::LevenbergMarquardtParams>> optimizer(graph_, initial_estimates_);
    gtsam::Values optimized_estimates;
    gtsam::Vector factor_weights;
    try {
        gtsam::GncOptimizer<gtsam::GncParams<gtsam::LevenbergMarquardtParams>> optimizer(graph_, unoptimized_estimates_,
                                                                                         gnc_params);
        optimized_estimates = optimizer.optimize();
        factor_weights = optimizer.getWeights();
    } catch (const std::exception& e) {
        Logger::error("GNC threw an exception, keeping the previous estimates: " + std::string(e.what()));

        // Reset as a successful optimization would, so the triggers in process_frames() do not retry straight away.
        num_new_loop_closures_ = 0;
        latest_optimization_time_ = std::chrono::steady_clock::now();
        return;
    }

    log_rejected_factors(factor_weights);

    {
        // gtsam::Values has no move assignment, so swap the result in rather than copy it, and let the
        // previous one die with the local once the lock is released.
        std::scoped_lock<std::mutex> lock(graph_mutex_);

        optimized_estimates_.swap(optimized_estimates);

        // Same result as optimized_estimates_, but updated incrementally with new lidar odometry
        // poses transformed into the map frame, to provide a more up-to-date estimate for visualization.
        online_estimates_ = optimized_estimates_;
    }

    // online_poses_ = extract_online_poses();
    // loop_closure_edges_ = extract_loop_closure_edges();

    Logger::info_important("Pose graph optimized.");

    // Very important: The map->odom transform is used to correct lidar odometry drift
    update_map_to_odom_transform();

    update_frame_poses();
    generate_optimized_vdb_map();

    num_new_loop_closures_ = 0;                                    // Reset the counter after optimization
    latest_optimization_time_ = std::chrono::steady_clock::now();  // Update the time of the latest optimization
}

void GraphOptimizer::log_rejected_factors(const gtsam::Vector& factor_weights) const {
    // GNC's TLS loss drives the weight of each loop closure and degraded odometry edge towards 0 or 1. We count a
    // weight below 0.5 as rejected.
    size_t num_rejected_loop_closures = 0;
    for (const size_t factor_index : loop_closure_factors_) {
        if (factor_weights(factor_index) < 0.5) {
            num_rejected_loop_closures++;
            Logger::info("GNC rejected the loop closure " + describe_edge(factor_index) + ".");
        }
    }

    size_t num_rejected_degraded_edges = 0;
    for (const size_t factor_index : degraded_odometry_factors_) {
        if (factor_weights(factor_index) < 0.5) {
            num_rejected_degraded_edges++;
            Logger::warn("GNC rejected the degraded odometry edge " + describe_edge(factor_index) + ".");
        }
    }

    Logger::info_important("GNC rejected " + std::to_string(num_rejected_loop_closures) + " of " +
                           std::to_string(loop_closure_factors_.size()) + " loop closures and " +
                           std::to_string(num_rejected_degraded_edges) + " of " +
                           std::to_string(degraded_odometry_factors_.size()) +
                           " degraded odometry edges, and kept all " + std::to_string(known_inlier_factors_.size()) +
                           " known inliers (the prior and the other odometry edges).");
}

std::string GraphOptimizer::describe_edge(const size_t factor_index) const {
    const gtsam::KeyVector& keys = graph_.at(factor_index)->keys();
    return gtsam::Symbol(keys.front()).string() + " -> " + gtsam::Symbol(keys.back()).string();
}

void GraphOptimizer::save_results() {
    Timer timer(__func__, LogLevel::InfoImportant);

    // No graph_mutex_: runs on the frame-processing thread, the sole writer of graph_ and the estimates.
    gtsam::writeG2o(graph_, unoptimized_estimates_, "unoptimized_graph.g2o");
    write_keyframe_stamps();

    if (optimized_estimates_.empty()) {
        Logger::warn("No optimization has run, skipping the optimized graph write.");
    } else {
        gtsam::writeG2o(graph_, optimized_estimates_, "optimized_graph.g2o");
    }

    save_trajectories();

    // Save point cloud map to disk. This writes the voxelized snapshot, whose size is
    // bounded by the volume explored.
    // IMPORTANT: The snapshot is immutable once published, so take the pointer under map_mutex_ and
    // release the lock before the slow disk write.
    pcl::PointCloud<pcl::PointXYZ>::ConstPtr map_snapshot;
    {
        std::scoped_lock<std::mutex> lock(map_mutex_);
        map_snapshot = rebuild_map_snapshot();
    }
    if (map_snapshot->empty()) {
        // Shutting down before any frame was mapped. PCL treats an empty cloud as an error.
        Logger::warn("Map is empty, skipping the point cloud write.");
    } else {
        pcl::io::savePLYFileBinary("map_pcd.ply", *map_snapshot);
        Logger::info_important("Map saved to disk.");
    }

    // // Save VDB map to disk
    // openvdb::io::File vdb_file("map.vdb");
    // openvdb::GridPtrVec grids;
    // grids.push_back(map_vdb_);
    // vdb_file.write(grids);
    // vdb_file.close();
}

void GraphOptimizer::save_trajectories() {
    // Written as timestamped TUM trajectories so they can be scored against ground truth with evo.
    write_keyframe_trajectory(unoptimized_estimates_, unoptimized_trajectory_file_,
                              "mapper3d lidar odometry at keyframes, odom frame");

    if (optimized_estimates_.empty()) {
        Logger::warn("No optimization has run, skipping the optimized trajectory write.");
    } else {
        write_keyframe_trajectory(optimized_estimates_, optimized_trajectory_file_,
                                  "mapper3d optimized pose graph, one pose per keyframe, map frame");
    }
}

void GraphOptimizer::write_keyframe_trajectory(const gtsam::Values& estimates, const std::string& path,
                                               const std::string& description) {
    // No graph_mutex_: called from save_results() on the frame-processing thread, the sole
    // writer of the estimates and of keyframe_timestamps_.

    std::ofstream file = utils::open_tum_file(path, description);
    if (!file) {
        return;
    }

    // gtsam::Values iterates in key order, so the poses come out in keyframe order. Index the
    // timestamps by the symbol's index rather than by loop position, so the pose and its stamp
    // cannot drift apart if a key is ever missing.
    size_t num_poses_written = 0;
    for (const auto& key_value : estimates) {
        const gtsam::Key key = key_value.key;
        if (!estimates.exists<gtsam::Pose3>(key)) {
            continue;
        }

        const size_t pose_index = gtsam::Symbol(key).index();
        if (pose_index >= keyframe_timestamps_.size()) {
            Logger::warn("No timestamp for pose index " + std::to_string(pose_index) + ", skipping it in " + path);
            continue;
        }

        const gtsam::Pose3 pose = estimates.at<gtsam::Pose3>(key);
        utils::write_tum_pose(file, keyframe_timestamps_[pose_index],
                              Sophus::SE3d(pose.rotation().toQuaternion(), pose.translation()));
        num_poses_written++;
    }

    Logger::info_important("Wrote " + std::to_string(num_poses_written) + " poses to " + path);
}

void GraphOptimizer::write_keyframe_stamps() const {
    std::ofstream stamps_file("keyframe_stamps.txt");
    if (!stamps_file) {
        Logger::warn("Could not open keyframe_stamps.txt for writing.");
        return;
    }

    stamps_file << "# keyframe_id scan_id timestamp\n";
    // No processed_frames_mutex_: the frame-processing thread is the sole writer and it is the
    // thread running this.
    for (const Frame& frame : processed_frames_) {
        stamps_file << frame.keyframe_id << ' ' << frame.point_cloud_id << ' ' << std::fixed << std::setprecision(9)
                    << frame.timestamp << '\n';
    }
}

void GraphOptimizer::update_map_to_odom_transform() {
    // No graph_mutex_: optimize() calls this on the thread that owns every member read below.
    const gtsam::Symbol latest_keyframe_symbol('x', current_keyframe_index_ - 1);
    const gtsam::Pose3 latest_unoptimized_pose = unoptimized_estimates_.at<gtsam::Pose3>(latest_keyframe_symbol);
    const gtsam::Pose3 latest_optimized_pose = optimized_estimates_.at<gtsam::Pose3>(latest_keyframe_symbol);

    const gtsam::Pose3 pose_difference = latest_optimized_pose * latest_unoptimized_pose.inverse();
    const Sophus::SE3d pose_correction(pose_difference.rotation().toQuaternion(), pose_difference.translation());

    {
        std::scoped_lock<std::mutex> lock(map_to_odom_transform_mutex_);
        map_to_odom_transform_ = pose_correction;
    }
}

void GraphOptimizer::update_frame_poses() {
    std::scoped_lock<std::mutex> lock(processed_frames_mutex_);

    // Use the poses stored in the pose graph to update all poses in processed_frames_
    for (auto& frame : processed_frames_) {
        const gtsam::Symbol keyframe_symbol('x', frame.keyframe_id);
        if (optimized_estimates_.exists(keyframe_symbol)) {
            const gtsam::Pose3 optimized_pose = optimized_estimates_.at<gtsam::Pose3>(keyframe_symbol);
            frame.pose = Sophus::SE3d(optimized_pose.rotation().toQuaternion(), optimized_pose.translation());
        } else {
            Logger::warn("Warning: Pose for keyframe id " + std::to_string(frame.keyframe_id) +
                         " not found in optimized estimates.");
        }
    }
}

void GraphOptimizer::generate_optimized_vdb_map() {
    Timer timer(__func__, LogLevel::InfoImportant);
    Logger::info_important("Generating optimized VDB map from optimized poses...");

    std::scoped_lock<std::mutex, std::mutex> lock(map_mutex_, processed_frames_mutex_);
    map_vdb_->clear();

    for (const auto& frame : processed_frames_) {
        update_vdb_map(frame.downsampled_point_cloud, frame.pose);
    }

    map_changed_ = true;

    Logger::info_important("Optimized map generated.");
}

void GraphOptimizer::update_vdb_map(const pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud,
                                    const Sophus::SE3d& sensor_pose) {
    Timer timer(__func__, LogLevel::Debug);
    Logger::debug("Updating VDB map...");

    // IMPORTANT: Need to make sure we're using the optimized poses from the graph optimization (i.e. sensor pose in map
    // frame, not sensor pose in odom frame)

    // IMPORTANT: Create the accessor once, outside the loop. The accessor caches the
    // tree traversal path between nearby lookups.
    openvdb::BoolGrid::Accessor accessor = map_vdb_->getAccessor();

    for (const auto& point : point_cloud->points) {
        const Eigen::Vector3d point_in_sensor_frame(point.x, point.y, point.z);
        const Eigen::Vector3d point_in_map_frame = sensor_pose * point_in_sensor_frame;

        const openvdb::Vec3d point_in_map_frame_vdb(point_in_map_frame.x(), point_in_map_frame.y(),
                                                    point_in_map_frame.z());

        // Convert the world-space point coordinate to an index-space coordinate.
        const openvdb::Coord voxel_coord = openvdb::Coord::floor(map_vdb_->worldToIndex(point_in_map_frame_vdb));

        accessor.setValueOn(voxel_coord, true);
    }

    map_changed_ = true;

    Logger::debug("VDB map updated.");
}

pcl::PointCloud<pcl::PointXYZ>::ConstPtr GraphOptimizer::rebuild_map_snapshot() {
    // IMPORTANT: The caller must hold map_mutex_.
    if (!map_changed_ && map_snapshot_) {
        return map_snapshot_;
    }

    Timer timer(__func__, LogLevel::Debug);

    // Rebuild the snapshot from the VDB grid.
    auto map_snapshot = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    map_snapshot->reserve(static_cast<size_t>(map_vdb_->activeVoxelCount()));

    for (auto iter = map_vdb_->cbeginValueOn(); iter; ++iter) {
        const openvdb::Vec3d point = map_vdb_->indexToWorld(iter.getCoord());
        map_snapshot->emplace_back(static_cast<float>(point.x()), static_cast<float>(point.y()),
                                   static_cast<float>(point.z()));
    }

    map_snapshot->width = map_snapshot->size();
    map_snapshot->height = 1;  // Unorganized point cloud
    map_snapshot->is_dense = false;

    // Publish by pointer swap. Nothing writes to the snapshot after this point, so a
    // reader that took the pointer under map_mutex_ can keep using it once it releases
    // the mutex.
    map_snapshot_ = map_snapshot;
    map_changed_ = false;

    return map_snapshot_;
}

void GraphOptimizer::get_visualization_data(VisualizationData& data) {
    {
        std::scoped_lock<std::mutex> lock(graph_mutex_);

        data.poses = extract_online_poses();
        data.loop_closure_edges = extract_loop_closure_edges();
    }

    {
        std::scoped_lock<std::mutex> lock(map_mutex_);

        // TODO: Is there a more efficient way to publish the map for visualization?
        // Hand over the snapshot pointer rather than a copy of the map. The snapshot is
        // immutable once published, so the caller can keep it after the mutex is gone.
        data.map = rebuild_map_snapshot();
    }
}

std::vector<Sophus::SE3d> GraphOptimizer::extract_online_poses() {
    // IMPORTANT: The caller must hold graph_mutex_.

    std::vector<Sophus::SE3d> poses;
    poses.reserve(online_estimates_.size());

    for (const auto& key_value : online_estimates_) {
        const gtsam::Key key = key_value.key;
        if (!online_estimates_.exists<gtsam::Pose3>(key)) {
            continue;
        }
        const gtsam::Pose3 pose = online_estimates_.at<gtsam::Pose3>(key);
        poses.emplace_back(Sophus::SE3d(pose.rotation().toQuaternion(), pose.translation()));
    }

    return poses;
}

std::vector<std::pair<size_t, size_t>> GraphOptimizer::extract_loop_closure_edges() {
    // IMPORTANT: The caller must hold graph_mutex_.

    std::vector<std::pair<size_t, size_t>> edges;

    for (const auto& factor : graph_) {
        const auto between_factor = boost::dynamic_pointer_cast<gtsam::BetweenFactor<gtsam::Pose3>>(factor);
        if (!between_factor) {
            continue;
        }
        const auto key1 = gtsam::Symbol(between_factor->key1());
        const auto key2 = gtsam::Symbol(between_factor->key2());
        if (key1.chr() == 'x' && key2.chr() == 'x') {
            // Heuristic: Loop closure edges are those where the indices are not consecutive
            if (key2.index() > key1.index() + 1) {
                edges.emplace_back(key1.index(), key2.index());
            }
        }
    }
    return edges;
}

pcl::PointCloud<pcl::PointXYZ>::ConstPtr GraphOptimizer::get_map() {
    std::scoped_lock<std::mutex> lock(map_mutex_);
    return rebuild_map_snapshot();
}

gtsam::Values GraphOptimizer::get_unoptimized_poses() {
    std::scoped_lock<std::mutex> lock(graph_mutex_);
    return unoptimized_estimates_;
}

gtsam::Values GraphOptimizer::get_optimized_poses() {
    std::scoped_lock<std::mutex> lock(graph_mutex_);
    return optimized_estimates_;
}

// std::vector<Sophus::SE3d> GraphOptimizer::get_online_poses() {
//     std::scoped_lock<std::mutex> lock(graph_mutex_);
//     return online_poses_;
// }

// std::vector<std::pair<size_t, size_t>> GraphOptimizer::get_loop_closure_edges() {
//     std::scoped_lock<std::mutex> lock(graph_mutex_);
//     return loop_closure_edges_;
// }

gtsam::NonlinearFactorGraph GraphOptimizer::get_graph() {
    std::scoped_lock<std::mutex> lock(graph_mutex_);
    return graph_;
}

// std::vector<Eigen::Matrix<double, 6, 6>> GraphOptimizer::get_pose_covariances() {
//     std::scoped_lock<std::mutex> lock(graph_mutex_);

//     std::vector<Eigen::Matrix<double, 6, 6>> covariances;
//     covariances.reserve(optimized_estimates_.size());

//     gtsam::Marginals marginals(graph_, optimized_estimates_);

//     for (const auto& key_value : optimized_estimates_) {
//         const gtsam::Key key = key_value.key;
//         if (!optimized_estimates_.exists<gtsam::Pose3>(key)) {
//             continue;
//         }

//         // Compute the marginal covariance for the pose
//         const Eigen::Matrix<double, 6, 6> covariance = marginals.marginalCovariance(key);

//         covariances.push_back(covariance);
//     }

//     return covariances;
// }

Sophus::SE3d GraphOptimizer::map_to_odom_transform() {
    std::scoped_lock<std::mutex> lock(map_to_odom_transform_mutex_);
    return map_to_odom_transform_;
}
}  // namespace mapper3d