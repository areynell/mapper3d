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

#include "mapper3d/mapper3d.h"

struct VisualizationData {
    pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud;
    std::vector<Eigen::Vector3f> kiss_matcher_point_cloud;
    std::vector<Eigen::Vector3f> kiss_matcher_keypoint_cloud;
    Sophus::SE3d sensor_pose;
    rclcpp::Time timestamp;
};

struct Mapper3DNodeParams {
    int sub_queue_size = 1;
    int pub_queue_size = 1;

    std::string map_frame_id = "map";
    std::string odom_frame_id = "odom";
    std::string base_frame_id = "base_link";
    std::string sensor_frame_id = "velodyne";

    // std::string point_cloud_sub_topic = "/points_raw";
    std::string point_cloud_sub_topic = "/velodyne_points";
    std::string point_cloud_pub_topic = "/filtered_points";
    std::string keypoint_pub_topic = "/keypoints";
    std::string pose_trajectory_pub_topic = "/pose_trajectory";

    mapper3d::Mapper3DParams mapper3d_params;
};

class Mapper3DNode : public rclcpp::Node {
   public:
    Mapper3DNode(const Mapper3DNodeParams &params);

   private:
    void pointcloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);

    void publish_point_cloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud, const std::string &frame_id,
                             const rclcpp::Time &timestamp);

    void publish_keypoint_cloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud, const std::string &frame_id,
                                const rclcpp::Time &timestamp);

    void publish_pose_trajectory(const Sophus::SE3d &pose, const std::string &frame_id, const rclcpp::Time &timestamp);

    void broadcast_transform(const Sophus::SE3d &transform, const std::string &frame_id,
                             const std::string &child_frame_id, const rclcpp::Time &timestamp);

    void enqueue_visualization_data(const rclcpp::Time &timestamp);

    void visualization_thread_func();

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr point_cloud_sub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr point_cloud_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr keypoint_pub_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pose_trajectory_pub_;

    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

    std::string map_frame_id_;
    std::string odom_frame_id_;
    std::string base_frame_id_;
    std::string sensor_frame_id_;

    mapper3d::Mapper3D mapper3d_;

    std::thread viz_thread_;
    std::mutex viz_mutex_;
    std::condition_variable viz_cv_;
    size_t max_viz_queue_size_;  // Prevent unlimited backlog
    bool run_viz_thread_;
    std::queue<std::shared_ptr<VisualizationData>> viz_queue_;

    nav_msgs::msg::Path pose_trajectory_;

    Sophus::SE3d map_to_sensor_transform_;

    rclcpp::Time prev_timestamp_;
};
