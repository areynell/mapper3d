#include "mapper3d/graph_optimizer.h"

#include <gtsam/nonlinear/GncOptimizer.h>
#include <gtsam/nonlinear/GncParams.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/dataset.h>
#include <pcl/io/ply_io.h>

#include <mutex>

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

    current_pose_index_ = 0;
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

    // Initialize point cloud map
    map_pcd_ = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    map_pcd_->is_dense = false;  // Allow for NaN values in the point cloud
    map_pcd_->height = 1;        // Set height to 1 for an unorganized point cloud
    map_pcd_->width = 0;         // Initialize width

    // Initialize OpenVDB for volumetric map representation
    openvdb::initialize();
    map_voxel_size_ = 0.05F;  // Adjust voxel size as needed
    map_vdb_ = openvdb::FloatGrid::create();
    map_vdb_->setTransform(openvdb::math::Transform::createLinearTransform(map_voxel_size_));
    map_vdb_->setGridClass(openvdb::GRID_LEVEL_SET);

    // Start processing frames in a separate thread
    frame_processing_thread_ = std::thread(&GraphOptimizer::process_frames, this);
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
    Frame frame;
    bool new_frame_available = false;

    while (!stop_processing_frames_) {
        Timer timer(__func__, LogLevel::Debug);

        new_frame_available = false;
        {
            std::unique_lock<std::mutex> lock(unprocessed_frames_mutex_);
            // Wait until:
            // 1) There's a frame in the queue, OR
            // 2) The stop_processing_frames_ is set, OR
            // 3) The optimization_trigger_interval has passed since the last optimization
            frame_processing_cv_.wait_for(lock, optimization_trigger_interval_,
                                          [this] { return !unprocessed_frames_.empty() || stop_processing_frames_; });

            // If stop is signaled and the queue is empty, exit the loop.
            if (stop_processing_frames_ && unprocessed_frames_.empty() && num_new_loop_closures_ == 0) {
                break;
            }

            // Process the frame if available
            if (!unprocessed_frames_.empty()) {
                Logger::debug("unprocessed_frames_.size() = " + std::to_string(unprocessed_frames_.size()));

                frame = unprocessed_frames_.front();
                unprocessed_frames_.pop();
                new_frame_available = true;
            }
        }

        if (new_frame_available) {
            add_odometry_constraint(frame.pose, frame.odometry_covariance);

            const auto loop_closure = loop_closure_detector_.detect_loop_closures(frame);
            if (loop_closure) {
                add_loop_closure_constraint(loop_closure->prev_id, loop_closure->id, loop_closure->relative_pose,
                                            loop_closure->covariance);
            }

            {
                // Incremental map updating
                std::scoped_lock<std::mutex> lock(map_mutex_);

                const auto sensor_pose_odom_frame = frame.pose;
                const auto sensor_pose_map_frame = map_to_odom_transform_ * sensor_pose_odom_frame;
                update_point_cloud_map(frame.point_cloud, sensor_pose_map_frame);
            }

            {
                std::scoped_lock<std::mutex> lock(processed_frames_mutex_);
                processed_frames_.emplace_back(frame);
            }
        }

        const auto current_time = std::chrono::steady_clock::now();
        const auto time_since_latest_optimization = current_time - latest_optimization_time_;
        const auto time_since_latest_frame_received = current_time - latest_frame_received_time_;

        Logger::info("Time since latest optimization: " +
                     std::to_string(std::chrono::duration<double>(time_since_latest_optimization).count()) +
                     " seconds.");

        // Perform optimization if there are more than N new loop closures since last optimization
        if ((num_new_loop_closures_ >= min_new_loop_closures_for_optimization_) &&
            (time_since_latest_optimization > optimization_trigger_interval_)) {
            optimize();
        }
        // Perform optimization if there are no unprocessed frames and enough time has passed since the last frame
        else if ((num_new_loop_closures_ > 0) && unprocessed_frames_.empty() &&
                 (time_since_latest_frame_received > optimization_trigger_interval_)) {
            optimize();
        }
    }
}

void GraphOptimizer::add_odometry_constraint(const Sophus::SE3d& pose,
                                             const Eigen::Matrix<double, 6, 6>& odometry_covariance) {
    std::scoped_lock<std::mutex> lock(graph_mutex_);

    const gtsam::Symbol current_pose_symbol('x', current_pose_index_);

    // if (incremental_optimization_) {
    //     const Sophus::SE3d pose_in_map_frame = map_to_odom_transform_ * pose;
    //     const gtsam::Pose3 current_pose(gtsam::Rot3(pose_in_map_frame.unit_quaternion()),
    //                                     gtsam::Point3(pose_in_map_frame.translation()));
    // }

    const gtsam::Pose3 current_pose(gtsam::Rot3(pose.unit_quaternion()), gtsam::Point3(pose.translation()));
    const auto odometry_noise_model = gtsam::noiseModel::Gaussian::Covariance(odometry_covariance);

    if (current_pose_index_ == 0) {
        // // Add prior factor for the first pose in the graph
        // const Eigen::Matrix<double, 6, 6> prior_covariance =
        //     Eigen::Matrix<double, 6, 6>::Identity() * 1.0e-9;  // Very small values
        // graph_.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(current_pose_symbol, current_pose, prior_covariance);

        graph_.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(current_pose_symbol, current_pose,
                                                                odometry_noise_model);
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

    // TODO: Is there a better way to create a pose graph that can be visualized live in RViz as it is being built?
    const Sophus::SE3d online_pose_in_map_frame = map_to_odom_transform_ * pose;
    const gtsam::Pose3 online_pose(gtsam::Rot3(online_pose_in_map_frame.unit_quaternion()),
                                   gtsam::Point3(online_pose_in_map_frame.translation()));
    online_estimates_.insert(current_pose_symbol, online_pose);

    current_pose_index_++;
}

void GraphOptimizer::add_loop_closure_constraint(size_t prev_pose_id, size_t pose_id, const Sophus::SE3d& relative_pose,
                                                 const Eigen::Matrix<double, 6, 6>& loop_closure_covariance) {
    std::scoped_lock<std::mutex> lock(graph_mutex_);

    const gtsam::Symbol prev_pose_symbol('x', prev_pose_id);
    const gtsam::Symbol current_pose_symbol('x', pose_id);

    const gtsam::Pose3 loop_closure_relative_pose(gtsam::Rot3(relative_pose.unit_quaternion()),
                                                  gtsam::Point3(relative_pose.translation()));
    const auto loop_closure_noise_model = gtsam::noiseModel::Gaussian::Covariance(loop_closure_covariance);

    // Add the loop closure factor
    graph_.emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(prev_pose_symbol, current_pose_symbol,
                                                              loop_closure_relative_pose, loop_closure_noise_model);

    num_new_loop_closures_++;
    num_loop_closures_++;
}

void GraphOptimizer::optimize() {
    {
        Timer timer(__func__, LogLevel::InfoImportant);
        std::scoped_lock<std::mutex> lock(graph_mutex_);

        Logger::info_important("Optimizing pose graph...");

        // gtsam::GncOptimizer<gtsam::GncParams<gtsam::LevenbergMarquardtParams>> optimizer(graph_, initial_estimates_);
        gtsam::GncOptimizer<gtsam::GncParams<gtsam::LevenbergMarquardtParams>> optimizer(graph_, initial_estimates_,
                                                                                         gnc_params_);

        optimized_estimates_ = optimizer.optimize();
        online_estimates_ = optimized_estimates_;

        Logger::info_important("Pose graph optimized.");
    }

    // Very important: The map->odom transform is used to correct lidar odometry drift
    update_map_to_odom_transform();

    update_frame_poses();
    // generate_optimized_vdb_map();
    generate_optimized_point_cloud_map();

    num_new_loop_closures_ = 0;                                    // Reset the counter after optimization
    latest_optimization_time_ = std::chrono::steady_clock::now();  // Update the time of the latest optimization

    // Save the graph to a file
    gtsam::writeG2o(graph_, initial_estimates_, "unoptimized_graph.g2o");
    gtsam::writeG2o(graph_, optimized_estimates_, "optimized_graph.g2o");

    // Save point cloud map to disk
    pcl::io::savePLYFileBinary("map_pcd.ply", *map_pcd_);

    // // Save VDB map to disk
    // openvdb::io::File vdb_file("map.vdb");
    // openvdb::GridPtrVec grids;
    // grids.push_back(map_vdb_);
    // vdb_file.write(grids);
    // vdb_file.close();

    Logger::info("Map saved to disk.");
}

void GraphOptimizer::update_map_to_odom_transform() {
    {
        std::scoped_lock<std::mutex> lock(graph_mutex_);

        const gtsam::Symbol latest_pose_symbol('x', current_pose_index_ - 1);
        const gtsam::Pose3 latest_unoptimized_pose = initial_estimates_.at<gtsam::Pose3>(latest_pose_symbol);
        const gtsam::Pose3 latest_optimized_pose = optimized_estimates_.at<gtsam::Pose3>(latest_pose_symbol);

        const gtsam::Pose3 pose_difference = latest_optimized_pose * latest_unoptimized_pose.inverse();
        const Sophus::SE3d pose_correction(pose_difference.rotation().toQuaternion(), pose_difference.translation());

        {
            std::scoped_lock<std::mutex> lock(map_to_odom_transform_mutex_);
            map_to_odom_transform_ = pose_correction;
        }
    }
}

void GraphOptimizer::update_frame_poses() {
    std::scoped_lock<std::mutex> lock(processed_frames_mutex_);

    // Use the poses stored in the pose graph to update all poses in processed_frames_
    for (auto& frame : processed_frames_) {
        const gtsam::Symbol pose_symbol('x', frame.id);
        if (optimized_estimates_.exists(pose_symbol)) {
            const gtsam::Pose3 optimized_pose = optimized_estimates_.at<gtsam::Pose3>(pose_symbol);
            frame.pose = Sophus::SE3d(optimized_pose.rotation().toQuaternion(), optimized_pose.translation());
        } else {
            Logger::warn("Warning: Pose for frame id " + std::to_string(frame.id) +
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
        update_vdb_map(frame.point_cloud, frame.pose);
    }

    Logger::info_important("Optimized map generated.");
}

void GraphOptimizer::update_vdb_map(const pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud,
                                    const Sophus::SE3d& sensor_pose) {
    Timer timer(__func__, LogLevel::Debug);
    Logger::debug("Updating VDB map...");

    // IMPORTANT: Need to make sure we're using the optimized poses from the graph optimization (i.e. sensor pose in map
    // frame, not sensor pose in odom frame)

    for (const auto& point : point_cloud->points) {
        const Eigen::Vector3d point_in_sensor_frame(point.x, point.y, point.z);
        const Eigen::Vector3d point_in_map_frame = sensor_pose * point_in_sensor_frame;

        const openvdb::Vec3d point_in_map_frame_vdb(point_in_map_frame.x(), point_in_map_frame.y(),
                                                    point_in_map_frame.z());

        // Convert the world-space point coordinate to an index-space coordinate.
        const openvdb::Coord voxel_coord = openvdb::Coord::floor(map_vdb_->worldToIndex(point_in_map_frame_vdb));

        // Access the voxel and set its value (e.g., to 1.0f to indicate occupancy)
        openvdb::FloatGrid::Accessor accessor = map_vdb_->getAccessor();
        accessor.setValue(voxel_coord, 100.0f);
    }

    Logger::debug("VDB map updated.");
}

void GraphOptimizer::generate_optimized_point_cloud_map() {
    Timer timer(__func__, LogLevel::InfoImportant);
    Logger::info_important("Generating optimized point cloud map from optimized poses...");

    std::scoped_lock<std::mutex, std::mutex> lock(map_mutex_, processed_frames_mutex_);
    map_pcd_->clear();

    // TODO: Generate point cloud map using full resolution point clouds
    for (const auto& frame : processed_frames_) {
        update_point_cloud_map(frame.point_cloud, frame.pose);
    }

    Logger::info_important("Optimized point cloud map generated.");
}

void GraphOptimizer::update_point_cloud_map(const pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud,
                                            const Sophus::SE3d& sensor_pose) {
    Timer timer(__func__, LogLevel::Debug);
    Logger::debug("Updating point cloud map...");

    // IMPORTANT: Need to make sure we're using the optimized poses from the graph optimization (i.e. sensor pose in map
    // frame, not sensor pose in odom frame)

    // Transform the point cloud from sensor frame to map frame
    pcl::PointCloud<pcl::PointXYZ> transformed_cloud;
    pcl::transformPointCloud(*point_cloud, transformed_cloud, sensor_pose.matrix().cast<float>());
    // Append the transformed cloud to the map point cloud
    *map_pcd_ += transformed_cloud;

    Logger::debug("Point cloud map updated.");
}

pcl::PointCloud<pcl::PointXYZ> GraphOptimizer::get_map() {
    std::scoped_lock<std::mutex> lock(map_mutex_);
    return *map_pcd_;
}

gtsam::Values GraphOptimizer::get_unoptimized_poses() {
    std::scoped_lock<std::mutex> lock(graph_mutex_);
    return initial_estimates_;
}

gtsam::Values GraphOptimizer::get_optimized_poses() {
    std::scoped_lock<std::mutex> lock(graph_mutex_);
    return optimized_estimates_;
}

gtsam::Values GraphOptimizer::get_online_poses() {
    std::scoped_lock<std::mutex> lock(graph_mutex_);
    return online_estimates_;
}

gtsam::NonlinearFactorGraph GraphOptimizer::get_graph() {
    std::scoped_lock<std::mutex> lock(graph_mutex_);
    return graph_;
}

Sophus::SE3d GraphOptimizer::map_to_odom_transform() {
    std::scoped_lock<std::mutex> lock(map_to_odom_transform_mutex_);
    return map_to_odom_transform_;
}
}  // namespace mapper3d