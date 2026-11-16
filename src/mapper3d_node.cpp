#include "mapper3d/mapper3d_node.h"

#include <gtsam/slam/BetweenFactor.h>
#include <pcl_conversions/pcl_conversions.h>
#include <tf2/convert.h>

#include <cstdint>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <mutex>
#include <rclcpp/logging.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/time.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <string>
#include <vector>

#include "mapper3d/common.h"
#include "mapper3d/logging.h"
#include "mapper3d/utils.h"

Mapper3DNode::Mapper3DNode(const rclcpp::NodeOptions &options)
    : Node("mapper3d_node", options),
      params_(load_parameters()),
      mapper3d_(params_.mapper3d_params),
      run_visualization_thread_(true),
      new_data_available_(false),
      map_publish_interval_(params_.map_publish_interval),
      //   map_to_sensor_transform_(Sophus::SE3d()),
      prev_timestamp_(rclcpp::Time(0, 0, RCL_ROS_TIME)) {
    RCLCPP_INFO(this->get_logger(), "Initializing Mapper3DNode...");

    point_cloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        params_.point_cloud_sub_topic, params_.sub_queue_size,
        std::bind(&Mapper3DNode::pointcloud_callback, this, std::placeholders::_1));

    pose_trajectory_pub_ =
        this->create_publisher<nav_msgs::msg::Path>(params_.pose_trajectory_pub_topic, params_.pub_queue_size);

    loop_closure_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(params_.loop_closure_pub_topic,
                                                                                     params_.pub_queue_size);

    map_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(params_.map_pub_topic, params_.pub_queue_size);

    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(this);

    visualization_wait_timeout_ = std::chrono::duration<double>(10.0);  // 10 seconds
    visualization_thread_ = std::thread(&Mapper3DNode::visualization_thread_func, this);

    RCLCPP_INFO(this->get_logger(), "Mapper3DNode initialized.");
}

Mapper3DNodeParams Mapper3DNode::load_parameters() {
    Mapper3DNodeParams params;
    load_node_parameters(params);
    return params;
}

void Mapper3DNode::load_node_parameters(Mapper3DNodeParams &node_params) {
    node_params.point_cloud_sub_topic =
        declare_parameter("node.point_cloud_sub_topic", node_params.point_cloud_sub_topic);
    node_params.pose_trajectory_pub_topic =
        declare_parameter("node.pose_trajectory_pub_topic", node_params.pose_trajectory_pub_topic);
    node_params.loop_closure_pub_topic =
        declare_parameter("node.loop_closure_pub_topic", node_params.loop_closure_pub_topic);
    node_params.map_pub_topic = declare_parameter("node.map_pub_topic", node_params.map_pub_topic);
    node_params.sub_queue_size = declare_parameter("node.sub_queue_size", node_params.sub_queue_size);
    node_params.pub_queue_size = declare_parameter("node.pub_queue_size", node_params.pub_queue_size);
    node_params.map_frame_id = declare_parameter("node.map_frame_id", node_params.map_frame_id);
    node_params.odom_frame_id = declare_parameter("node.odom_frame_id", node_params.odom_frame_id);
    node_params.sensor_frame_id = declare_parameter("node.sensor_frame_id", node_params.sensor_frame_id);
    node_params.map_publish_interval = declare_parameter("node.map_publish_interval", node_params.map_publish_interval);
    node_params.map_publish_voxel_size =
        declare_parameter("node.map_publish_voxel_size", node_params.map_publish_voxel_size);

    load_front_end_parameters(node_params.mapper3d_params);
}

void Mapper3DNode::load_front_end_parameters(mapper3d::Mapper3DParams &front_end) {
    // KISS-Matcher keypoint matching needs them to be equal.
    front_end.voxel_size = declare_parameter("voxel_size", front_end.voxel_size);
    front_end.graph_optimizer_params.loop_closure_detector_params.voxel_size = front_end.voxel_size;

    front_end.z_height_threshold = declare_parameter("front_end.z_height_threshold", front_end.z_height_threshold);
    front_end.min_sensor_distance = declare_parameter("front_end.min_sensor_distance", front_end.min_sensor_distance);
    front_end.keyframe_distance_threshold =
        declare_parameter("front_end.keyframe_distance_threshold", front_end.keyframe_distance_threshold);
    front_end.keyframe_rotation_threshold = mapper3d::utils::deg2rad(declare_parameter(
        "front_end.keyframe_rotation_threshold_deg", mapper3d::utils::rad2deg(front_end.keyframe_rotation_threshold)));
    front_end.lidar_odometry_trajectory_file =
        declare_parameter("front_end.lidar_odometry_trajectory_file", front_end.lidar_odometry_trajectory_file);
    front_end.online_trajectory_file =
        declare_parameter("front_end.online_trajectory_file", front_end.online_trajectory_file);
    front_end.x_odom_uncertainty = declare_parameter("front_end.x_odom_uncertainty", front_end.x_odom_uncertainty);
    front_end.y_odom_uncertainty = declare_parameter("front_end.y_odom_uncertainty", front_end.y_odom_uncertainty);
    front_end.z_odom_uncertainty = declare_parameter("front_end.z_odom_uncertainty", front_end.z_odom_uncertainty);
    front_end.roll_odom_uncertainty = mapper3d::utils::deg2rad(declare_parameter(
        "front_end.roll_odom_uncertainty_deg", mapper3d::utils::rad2deg(front_end.roll_odom_uncertainty)));
    front_end.pitch_odom_uncertainty = mapper3d::utils::deg2rad(declare_parameter(
        "front_end.pitch_odom_uncertainty_deg", mapper3d::utils::rad2deg(front_end.pitch_odom_uncertainty)));
    front_end.yaw_odom_uncertainty = mapper3d::utils::deg2rad(declare_parameter(
        "front_end.yaw_odom_uncertainty_deg", mapper3d::utils::rad2deg(front_end.yaw_odom_uncertainty)));

    load_local_registration_parameters(front_end.local_reg_params);
    load_graph_optimizer_parameters(front_end.graph_optimizer_params);
}

void Mapper3DNode::load_local_registration_parameters(mapper3d::LocalRegistrationParams &local_registration) {
    local_registration.num_threads =
        declare_parameter("local_registration.num_threads", local_registration.num_threads);
    local_registration.num_neighbors_for_covariance = declare_parameter(
        "local_registration.num_neighbors_for_covariance", local_registration.num_neighbors_for_covariance);
    local_registration.wide_search_point_clouds_after_reset =
        declare_parameter<int64_t>("local_registration.wide_search_point_clouds_after_reset",
                                   local_registration.wide_search_point_clouds_after_reset);
    local_registration.dropped_point_cloud_interval = declare_parameter(
        "local_registration.dropped_point_cloud_interval", local_registration.dropped_point_cloud_interval);
    local_registration.lru_horizon =
        declare_parameter<int64_t>("local_registration.lru_horizon", local_registration.lru_horizon);
    local_registration.lru_clear_cycle =
        declare_parameter<int64_t>("local_registration.lru_clear_cycle", local_registration.lru_clear_cycle);
    local_registration.min_inlier_ratio =
        declare_parameter("local_registration.min_inlier_ratio", local_registration.min_inlier_ratio);
    local_registration.max_error_per_inlier =
        declare_parameter("local_registration.max_error_per_inlier", local_registration.max_error_per_inlier);
    local_registration.max_consecutive_failures = declare_parameter<int64_t>(
        "local_registration.max_consecutive_failures", local_registration.max_consecutive_failures);
    local_registration.max_plausible_sensor_speed = declare_parameter("local_registration.max_plausible_sensor_speed",
                                                                      local_registration.max_plausible_sensor_speed);
}

void Mapper3DNode::load_graph_optimizer_parameters(mapper3d::GraphOptimizerParams &graph_optimizer) {
    graph_optimizer.max_iterations =
        declare_parameter("graph_optimizer.max_iterations", graph_optimizer.max_iterations);
    graph_optimizer.map_voxel_size =
        declare_parameter("graph_optimizer.map_voxel_size", graph_optimizer.map_voxel_size);
    graph_optimizer.prior_rotation_sigma =
        declare_parameter("graph_optimizer.prior_rotation_sigma_rad", graph_optimizer.prior_rotation_sigma);
    graph_optimizer.prior_translation_sigma =
        declare_parameter("graph_optimizer.prior_translation_sigma", graph_optimizer.prior_translation_sigma);
    graph_optimizer.optimization_trigger_interval = declare_parameter("graph_optimizer.optimization_trigger_interval",
                                                                      graph_optimizer.optimization_trigger_interval);
    graph_optimizer.min_new_loop_closures_for_optimization =
        declare_parameter<int64_t>("graph_optimizer.min_new_loop_closures_for_optimization",
                                   graph_optimizer.min_new_loop_closures_for_optimization);
    graph_optimizer.unoptimized_trajectory_file =
        declare_parameter("graph_optimizer.unoptimized_trajectory_file", graph_optimizer.unoptimized_trajectory_file);
    graph_optimizer.optimized_trajectory_file =
        declare_parameter("graph_optimizer.optimized_trajectory_file", graph_optimizer.optimized_trajectory_file);

    load_loop_closure_detector_parameters(graph_optimizer.loop_closure_detector_params);
}

void Mapper3DNode::load_loop_closure_detector_parameters(mapper3d::LoopClosureDetectorParams &loop_closure) {
    loop_closure.max_correspondence_distance = declare_parameter("loop_closure_detector.max_correspondence_distance",
                                                                 loop_closure.max_correspondence_distance);
    loop_closure.num_threads = declare_parameter("loop_closure_detector.num_threads", loop_closure.num_threads);
    loop_closure.min_final_inliers =
        declare_parameter<int64_t>("loop_closure_detector.min_final_inliers", loop_closure.min_final_inliers);
    loop_closure.min_inlier_ratio =
        declare_parameter("loop_closure_detector.min_inlier_ratio", loop_closure.min_inlier_ratio);
    loop_closure.x_loop_uncertainty =
        declare_parameter("loop_closure_detector.x_loop_uncertainty", loop_closure.x_loop_uncertainty);
    loop_closure.y_loop_uncertainty =
        declare_parameter("loop_closure_detector.y_loop_uncertainty", loop_closure.y_loop_uncertainty);
    loop_closure.z_loop_uncertainty =
        declare_parameter("loop_closure_detector.z_loop_uncertainty", loop_closure.z_loop_uncertainty);
    loop_closure.roll_loop_uncertainty =
        mapper3d::utils::deg2rad(declare_parameter("loop_closure_detector.roll_loop_uncertainty_deg",
                                                   mapper3d::utils::rad2deg(loop_closure.roll_loop_uncertainty)));
    loop_closure.pitch_loop_uncertainty =
        mapper3d::utils::deg2rad(declare_parameter("loop_closure_detector.pitch_loop_uncertainty_deg",
                                                   mapper3d::utils::rad2deg(loop_closure.pitch_loop_uncertainty)));
    loop_closure.yaw_loop_uncertainty = mapper3d::utils::deg2rad(declare_parameter(
        "loop_closure_detector.yaw_loop_uncertainty_deg", mapper3d::utils::rad2deg(loop_closure.yaw_loop_uncertainty)));
}

Mapper3DNode::~Mapper3DNode() {
    {
        std::scoped_lock<std::mutex> lock(visualization_mutex_);
        run_visualization_thread_ = false;
    }

    visualization_cv_.notify_all();

    if (visualization_thread_.joinable()) {
        visualization_thread_.join();
    }
}

void Mapper3DNode::pointcloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    Timer timer(__func__, LogLevel::InfoImportant);

    const rclcpp::Time msg_time(msg->header.stamp, RCL_ROS_TIME);
    const double time_diff = (msg_time - prev_timestamp_).seconds();
    RCLCPP_INFO(this->get_logger(), "Point cloud time difference: %.3f seconds", time_diff);
    if (msg_time <= prev_timestamp_) {
        RCLCPP_ERROR(this->get_logger(), "Received point cloud with non-increasing timestamp.");
        // exit(EXIT_FAILURE);
        // return;
    }
    prev_timestamp_ = msg_time;

    const auto cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    pcl::fromROSMsg(*msg, *cloud);

    mapper3d_.spin_once(cloud, msg_time.seconds());
    // map_to_sensor_transform_ = mapper3d_.sensor_pose();
    // map_to_sensor_transform_ = mapper3d_.sensor_pose_map_frame();

    const auto map_to_odom_transform = mapper3d_.sensor_pose_map_frame() * mapper3d_.sensor_pose_odom_frame().inverse();
    const auto odom_to_sensor_transform = mapper3d_.sensor_pose_odom_frame();
    broadcast_transform(map_to_odom_transform, params_.map_frame_id, params_.odom_frame_id, msg->header.stamp);
    broadcast_transform(odom_to_sensor_transform, params_.odom_frame_id, params_.sensor_frame_id, msg->header.stamp);
    // broadcast_transform(map_to_sensor_transform_, params_.map_frame_id, params_.sensor_frame_id, msg->header.stamp);

    notify_visualization_thread();

    RCLCPP_INFO(this->get_logger(), "--------------------------------------------------------------------------");
}

void Mapper3DNode::broadcast_transform(const Sophus::SE3d &transform, const std::string &frame_id,
                                       const std::string &child_frame_id, const rclcpp::Time &timestamp) {
    geometry_msgs::msg::TransformStamped tf_msg;
    tf_msg.header.stamp = timestamp;
    tf_msg.header.frame_id = frame_id;
    tf_msg.child_frame_id = child_frame_id;

    tf_msg.transform.translation.x = transform.translation().x();
    tf_msg.transform.translation.y = transform.translation().y();
    tf_msg.transform.translation.z = transform.translation().z();

    tf_msg.transform.rotation.x = transform.unit_quaternion().x();
    tf_msg.transform.rotation.y = transform.unit_quaternion().y();
    tf_msg.transform.rotation.z = transform.unit_quaternion().z();
    tf_msg.transform.rotation.w = transform.unit_quaternion().w();

    tf_broadcaster_->sendTransform(tf_msg);
}

void Mapper3DNode::notify_visualization_thread() {
    std::scoped_lock<std::mutex> lock(visualization_mutex_);
    new_data_available_ = true;
    visualization_cv_.notify_one();
}

void Mapper3DNode::visualization_thread_func() {
    while (true) {
        {
            std::unique_lock<std::mutex> lock(visualization_mutex_);
            visualization_cv_.wait(lock, [this] { return !run_visualization_thread_ || new_data_available_; });

            if (!run_visualization_thread_) {
                break;
            }

            // Reset data flag after being woken up
            new_data_available_ = false;
        }

        RCLCPP_INFO(this->get_logger(), "Processing visualization data...");
        Timer timer(__func__, LogLevel::InfoImportant);

        // Fetch a single, consistent snapshot of all data
        mapper3d::VisualizationData graph_vis_data;
        mapper3d_.get_graph_visualization_data(graph_vis_data);

        // Why does global_registration_point_cloud "smear" during turns? Is there a timestamp issue?
        // NOTE: it's not due to point cloud warping

        // TODO: Use a more appropriate timestamp for visualization
        const auto timestamp = this->now();

        publish_pose_trajectory(graph_vis_data.poses, timestamp);
        publish_loop_closures(graph_vis_data.poses, graph_vis_data.loop_closure_edges, timestamp);

        const auto current_time = std::chrono::steady_clock::now();
        if (current_time - prev_map_publish_time_ >= map_publish_interval_) {
            publish_map(graph_vis_data.map, timestamp);
            prev_map_publish_time_ = current_time;
        }
    }
}

void Mapper3DNode::publish_pose_trajectory(const std::vector<Sophus::SE3d> &poses, const rclcpp::Time &timestamp) {
    Timer timer(__func__, LogLevel::Debug);

    if (pose_trajectory_pub_->get_subscription_count() == 0) {
        return;
    }

    nav_msgs::msg::Path path;
    path.header.stamp = timestamp;
    path.header.frame_id = params_.map_frame_id;
    path.poses.reserve(poses.size());

    for (const auto &pose : poses) {
        geometry_msgs::msg::PoseStamped pose_stamped;
        pose_stamped.header.stamp = path.header.stamp;
        pose_stamped.header.frame_id = path.header.frame_id;

        pose_stamped.pose.position.x = pose.translation().x();
        pose_stamped.pose.position.y = pose.translation().y();
        pose_stamped.pose.position.z = pose.translation().z();

        const auto quat = pose.unit_quaternion();
        pose_stamped.pose.orientation.x = quat.x();
        pose_stamped.pose.orientation.y = quat.y();
        pose_stamped.pose.orientation.z = quat.z();
        pose_stamped.pose.orientation.w = quat.w();

        path.poses.emplace_back(pose_stamped);
    }

    assert(path.poses.size() == values.size());

    // RCLCPP_DEBUG(this->get_logger(), "Publishing pose graph");
    pose_trajectory_pub_->publish(path);
}

void Mapper3DNode::publish_loop_closures(const std::vector<Sophus::SE3d> &poses,
                                         const std::vector<std::pair<size_t, size_t>> &loop_closure_edges,
                                         const rclcpp::Time &timestamp) {
    Timer timer(__func__, LogLevel::Debug);

    if (loop_closure_pub_->get_subscription_count() == 0) {
        return;
    }

    visualization_msgs::msg::MarkerArray loop_closures;
    auto &loop_closure = loop_closures.markers.emplace_back();

    loop_closure.header.frame_id = params_.map_frame_id;
    loop_closure.header.stamp = timestamp;
    loop_closure.ns = "loop_closures";
    loop_closure.id = 0;
    loop_closure.type = visualization_msgs::msg::Marker::LINE_LIST;
    loop_closure.action = visualization_msgs::msg::Marker::ADD;
    loop_closure.pose.orientation.w = 1.0;
    loop_closure.scale.x = 0.05;  // Line width
    loop_closure.color.r = 1.0;
    loop_closure.color.g = 0.0;
    loop_closure.color.b = 1.0;
    loop_closure.color.a = 0.8;

    for (const std::pair<size_t, size_t> &edge : loop_closure_edges) {
        const Sophus::SE3d &pose1 = poses[edge.first];
        const Sophus::SE3d &pose2 = poses[edge.second];

        geometry_msgs::msg::Point point1;
        point1.x = pose1.translation().x();
        point1.y = pose1.translation().y();
        point1.z = pose1.translation().z();

        geometry_msgs::msg::Point point2;
        point2.x = pose2.translation().x();
        point2.y = pose2.translation().y();
        point2.z = pose2.translation().z();

        loop_closure.points.emplace_back(point1);
        loop_closure.points.emplace_back(point2);
    }

    loop_closure_pub_->publish(loop_closures);
}

void Mapper3DNode::publish_map(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr &map, const rclcpp::Time &timestamp) {
    Timer timer(__func__, LogLevel::Debug);

    if (!map || map_pub_->get_subscription_count() == 0) {
        return;
    }

    // Downsample the map to reduce size. The snapshot is passed by pointer, so no copy
    // of the map is made here.
    pcl::PointCloud<pcl::PointXYZ>::Ptr downsampled_map(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::VoxelGrid<pcl::PointXYZ> voxel_grid;
    const auto voxel_size = static_cast<float>(params_.map_publish_voxel_size);
    voxel_grid.setInputCloud(map);
    voxel_grid.setLeafSize(voxel_size, voxel_size, voxel_size);
    voxel_grid.filter(*downsampled_map);

    sensor_msgs::msg::PointCloud2 map_msg;
    pcl::toROSMsg(*downsampled_map, map_msg);
    map_msg.header.frame_id = params_.map_frame_id;
    map_msg.header.stamp = timestamp;

    map_pub_->publish(map_msg);

    RCLCPP_INFO_STREAM(this->get_logger(), "Published map with " << map->size() << " points.");
}

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<Mapper3DNode>());
    rclcpp::shutdown();
    return 0;
}