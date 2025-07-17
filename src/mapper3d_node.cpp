#include "mapper3d/mapper3d_node.h"

#include <gtsam/slam/BetweenFactor.h>
#include <pcl_conversions/pcl_conversions.h>
#include <tf2/convert.h>

#include <cstdlib>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <mutex>
#include <rclcpp/logging.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include "mapper3d/logging.h"
#include "mapper3d/utils.h"

Mapper3DNode::Mapper3DNode(const Mapper3DNodeParams &params)
    : Node("mapper3d_node"),
      mapper3d_(params.mapper3d_params),
      map_frame_id_(params.map_frame_id),
      odom_frame_id_(params.odom_frame_id),
      base_frame_id_(params.base_frame_id),
      sensor_frame_id_(params.sensor_frame_id),
      run_viz_thread_(true),
      max_viz_queue_size_(5),
      map_to_sensor_transform_(Sophus::SE3d()),
      prev_timestamp_(rclcpp::Time(0, 0, RCL_ROS_TIME)) {
    RCLCPP_INFO(this->get_logger(), "Initializing Mapper3DNode...");

    point_cloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        params.point_cloud_sub_topic, params.sub_queue_size,
        std::bind(&Mapper3DNode::pointcloud_callback, this, std::placeholders::_1));

    point_cloud_pub_ =
        this->create_publisher<sensor_msgs::msg::PointCloud2>(params.point_cloud_pub_topic, params.pub_queue_size);

    keypoint_pub_ =
        this->create_publisher<sensor_msgs::msg::PointCloud2>(params.keypoint_pub_topic, params.pub_queue_size);

    pose_trajectory_pub_ =
        this->create_publisher<nav_msgs::msg::Path>(params.pose_trajectory_pub_topic, params.pub_queue_size);

    loop_closure_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(params.loop_closure_pub_topic,
                                                                                     params.pub_queue_size);

    map_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(params.map_pub_topic, params.pub_queue_size);

    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(this);

    viz_thread_ = std::thread(&Mapper3DNode::visualization_thread_func, this);

    visualization_wait_timeout_ = std::chrono::duration<double>(10.0);  // 10 seconds

    RCLCPP_INFO(this->get_logger(), "Mapper3DNode initialized.");
}

void Mapper3DNode::pointcloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    Timer timer(__func__, LogLevel::Debug);

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

    mapper3d_.spin_once(cloud);
    // map_to_sensor_transform_ = mapper3d_.sensor_pose();
    // map_to_sensor_transform_ = mapper3d_.sensor_pose_map_frame();

    const auto map_to_odom_transform = mapper3d_.sensor_pose_map_frame() * mapper3d_.sensor_pose_odom_frame().inverse();
    const auto odom_to_sensor_transform = mapper3d_.sensor_pose_odom_frame();
    broadcast_transform(map_to_odom_transform, map_frame_id_, odom_frame_id_, msg->header.stamp);
    broadcast_transform(odom_to_sensor_transform, odom_frame_id_, sensor_frame_id_, msg->header.stamp);
    // broadcast_transform(map_to_sensor_transform_, map_frame_id_, sensor_frame_id_, msg->header.stamp);

    enqueue_visualization_data(msg_time);

    Logger::info("--------------------------------------------------------------------------");
}

// TODO: Should we remove this function and just publish directly from the visualization thread?
void Mapper3DNode::enqueue_visualization_data(const rclcpp::Time &timestamp) {
    // TODO: Do we really need a visualization data struct?
    // Instantiate visualization data struct
    auto viz_data = std::make_shared<VisualizationData>();
    viz_data->point_cloud = mapper3d_.point_cloud();
    viz_data->kiss_matcher_point_cloud = mapper3d_.kiss_matcher_point_cloud();
    viz_data->kiss_matcher_keypoint_cloud = mapper3d_.kiss_matcher_keypoint_cloud();
    viz_data->sensor_pose = mapper3d_.sensor_pose_odom_frame();
    viz_data->timestamp = timestamp;

    // Add visualization data to queue with lock
    {
        std::scoped_lock<std::mutex> lock(viz_mutex_);
        // Prevent queue from growing too large if visualization can't keep up
        while (viz_queue_.size() >= max_viz_queue_size_) {
            viz_queue_.pop();
        }
        viz_queue_.push(viz_data);
    }
    viz_cv_.notify_one();
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

void Mapper3DNode::visualization_thread_func() {
    while (run_viz_thread_) {
        std::shared_ptr<VisualizationData> viz_data;

        {
            std::unique_lock<std::mutex> lock(viz_mutex_);
            viz_cv_.wait_for(lock, visualization_wait_timeout_,
                             [this]() { return !viz_queue_.empty() || !run_viz_thread_; });

            if (!run_viz_thread_) {
                break;
            }

            if (!viz_queue_.empty()) {
                viz_data = viz_queue_.front();
                viz_queue_.pop();
            }
        }

        // TODO: Should we remove this check?
        if (viz_data) {
            auto kiss_matcher_point_cloud = mapper3d::utils::point_vec_to_cloud(viz_data->kiss_matcher_point_cloud);
            auto kiss_matcher_keypoints = mapper3d::utils::point_vec_to_cloud(viz_data->kiss_matcher_keypoint_cloud);

            // publish_point_cloud(viz_data->point_cloud, sensor_frame_id_, viz_data->timestamp);
            // Why does kiss_matcher_point_cloud "smear" during turns? Is there a timestamp issue? NOTE: it's not due to
            // point cloud warping
            // publish_point_cloud(kiss_matcher_point_cloud, sensor_frame_id_, viz_data->timestamp);
            publish_keypoint_cloud(kiss_matcher_keypoints, sensor_frame_id_, viz_data->timestamp);
        }
        publish_pose_trajectory();
        publish_loop_closures();
        // publish_map();
    }
}

void Mapper3DNode::publish_point_cloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud,
                                       const std::string &frame_id, const rclcpp::Time &timestamp) {
    sensor_msgs::msg::PointCloud2 point_cloud_msg;
    pcl::toROSMsg(*point_cloud, point_cloud_msg);
    point_cloud_msg.header.frame_id = frame_id;
    point_cloud_msg.header.stamp = timestamp;

    point_cloud_pub_->publish(point_cloud_msg);
}

void Mapper3DNode::publish_keypoint_cloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud,
                                          const std::string &frame_id, const rclcpp::Time &timestamp) {
    sensor_msgs::msg::PointCloud2 point_cloud_msg;
    pcl::toROSMsg(*point_cloud, point_cloud_msg);
    point_cloud_msg.header.frame_id = frame_id;
    point_cloud_msg.header.stamp = timestamp;

    keypoint_pub_->publish(point_cloud_msg);
}

void Mapper3DNode::publish_pose_trajectory() {
    if (pose_trajectory_pub_->get_subscription_count() > 0) {
        const gtsam::Values &graph = mapper3d_.get_online_poses();

        nav_msgs::msg::Path path;
        path.header.stamp = this->now();
        path.header.frame_id = map_frame_id_;
        path.poses.reserve(graph.size());

        for (const auto &[key, value] : graph) {
            gtsam::Pose3 pose;
            if (graph.exists<gtsam::Pose3>(key)) {
                pose = graph.at<gtsam::Pose3>(key);
            } else {
                Logger::warn("Warning: Pose for key " + std::to_string(key) + " not found in graph.");
                continue;
            }

            geometry_msgs::msg::PoseStamped pose_stamped;
            pose_stamped.header.stamp = path.header.stamp;
            pose_stamped.header.frame_id = path.header.frame_id;

            pose_stamped.pose.position.x = pose.translation().x();
            pose_stamped.pose.position.y = pose.translation().y();
            pose_stamped.pose.position.z = pose.translation().z();

            const auto quat = pose.rotation().toQuaternion();
            pose_stamped.pose.orientation.x = quat.x();
            pose_stamped.pose.orientation.y = quat.y();
            pose_stamped.pose.orientation.z = quat.z();
            pose_stamped.pose.orientation.w = quat.w();

            path.poses.emplace_back(pose_stamped);
        }

        assert(path.poses.size() == graph.size());

        // Logger::debug("Publishing pose graph");
        pose_trajectory_pub_->publish(path);
    }
}

void Mapper3DNode::publish_loop_closures() {
    if (loop_closure_pub_->get_subscription_count() > 0) {
        const auto &poses = mapper3d_.get_online_poses();
        const auto &factors = mapper3d_.get_graph();

        visualization_msgs::msg::MarkerArray loop_closures;
        auto &loop_closure = loop_closures.markers.emplace_back();

        loop_closure.header.frame_id = map_frame_id_;
        loop_closure.header.stamp = this->now();
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

        for (const auto &factor : factors) {
            const auto between_factor = boost::dynamic_pointer_cast<gtsam::BetweenFactor<gtsam::Pose3>>(factor);
            if (!between_factor) {
                continue;
            }

            const gtsam::Key key1 = between_factor->key1();
            const gtsam::Key key2 = between_factor->key2();

            // Heuristic: loop closures connect non-consecutive frames
            if (gtsam::Symbol(key2).index() > gtsam::Symbol(key1).index() + 1) {
                if (poses.exists(key1) && poses.exists(key2)) {
                    const gtsam::Pose3 pose1 = poses.at<gtsam::Pose3>(key1);
                    const gtsam::Pose3 pose2 = poses.at<gtsam::Pose3>(key2);

                    geometry_msgs::msg::Point position1;
                    position1.x = pose1.translation().x();
                    position1.y = pose1.translation().y();
                    position1.z = pose1.translation().z();

                    geometry_msgs::msg::Point position2;
                    position2.x = pose2.translation().x();
                    position2.y = pose2.translation().y();
                    position2.z = pose2.translation().z();

                    loop_closure.points.push_back(position1);
                    loop_closure.points.push_back(position2);
                }
            }
        }
        loop_closure_pub_->publish(loop_closures);
    }
}

void Mapper3DNode::publish_map() {
    Timer timer(__func__, LogLevel::Debug);

    if (map_pub_->get_subscription_count() > 0) {
        const auto map = mapper3d_.get_map();

        sensor_msgs::msg::PointCloud2 map_msg;
        pcl::toROSMsg(map, map_msg);
        map_msg.header.frame_id = map_frame_id_;
        map_msg.header.stamp = this->now();

        map_pub_->publish(map_msg);

        RCLCPP_INFO_STREAM(this->get_logger(), "Published map with " << map.size() << " points.");
    }
}

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);

    Mapper3DNodeParams params;
    auto node = std::make_shared<Mapper3DNode>(params);

    rclcpp::spin(node);
    rclcpp::shutdown();

    return 0;
}