#pragma once

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>

#include <condition_variable>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <nav_msgs/msg/path.hpp>
#include <pcl/impl/point_types.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <string>
#include <visualization_msgs/msg/marker_array.hpp>

#include "mapper3d/mapper3d.h"

struct Mapper3DNodeParams {
    int sub_queue_size = 100;
    int pub_queue_size = 1;

    std::string map_frame_id = "map";
    std::string odom_frame_id = "odom";
    std::string base_frame_id = "base_link";
    std::string sensor_frame_id = "velodyne";

    // std::string point_cloud_sub_topic = "/points_raw";
    std::string point_cloud_sub_topic = "/velodyne_points";
    std::string pose_trajectory_pub_topic = "/pose_trajectory";
    std::string loop_closure_pub_topic = "/loop_closures";
    std::string map_pub_topic = "/map";

    mapper3d::Mapper3DParams mapper3d_params;
};

class Mapper3DNode : public rclcpp::Node {
   public:
    Mapper3DNode(const Mapper3DNodeParams &params);
    ~Mapper3DNode();

   private:
    void pointcloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);

    void publish_pose_trajectory(const std::vector<Sophus::SE3d> &poses, const rclcpp::Time &timestamp);

    void publish_loop_closures(const std::vector<Sophus::SE3d> &poses,
                               const std::vector<std::pair<size_t, size_t>> &loop_closure_edges,
                               const rclcpp::Time &timestamp);

    void publish_map(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr &map, const rclcpp::Time &timestamp);

    void broadcast_transform(const Sophus::SE3d &transform, const std::string &frame_id,
                             const std::string &child_frame_id, const rclcpp::Time &timestamp);

    void enqueue_visualization_data(const rclcpp::Time &timestamp);
    void notify_visualization_thread();

    void visualization_thread_func();

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr point_cloud_sub_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pose_trajectory_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr loop_closure_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr map_pub_;

    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

    std::string map_frame_id_;
    std::string odom_frame_id_;
    std::string base_frame_id_;
    std::string sensor_frame_id_;

    mapper3d::Mapper3D mapper3d_;

    bool run_visualization_thread_;
    std::thread visualization_thread_;
    std::mutex visualization_mutex_;
    std::condition_variable visualization_cv_;
    std::chrono::duration<double> visualization_wait_timeout_;
    bool new_data_available_;

    // Sophus::SE3d map_to_sensor_transform_;

    rclcpp::Time prev_timestamp_;
};
