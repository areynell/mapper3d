#pragma once

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>

#include <geometry_msgs/msg/transform_stamped.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <string>

#include "mapper3d/mapper3d.h"

struct Mapper3DNodeParams {
    int sub_queue_size = 100;
    int pub_queue_size = 100;

    std::string map_frame_id = "map";
    std::string odom_frame_id = "odom";
    std::string base_frame_id = "base_link";
    std::string sensor_frame_id = "velodyne";

    std::string point_cloud_sub_topic = "/points_raw";

    Mapper3DParams mapper3d_params;
};

class Mapper3DNode : public rclcpp::Node {
   public:
    Mapper3DNode(const Mapper3DNodeParams &params);

   private:
    void pointcloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);

    void broadcast_transform(const Sophus::SE3d &transform, const std::string &frame_id,
                             const std::string &child_frame_id, const rclcpp::Time &timestamp);

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr point_cloud_sub_;

    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

    std::string map_frame_id_;
    std::string odom_frame_id_;
    std::string base_frame_id_;
    std::string sensor_frame_id_;

    Mapper3D mapper3d_;
};
