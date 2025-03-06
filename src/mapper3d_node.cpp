#include "mapper3d/mapper3d_node.h"

#include <pcl_conversions/pcl_conversions.h>

#include <geometry_msgs/msg/transform_stamped.hpp>
#include <rclcpp/logging.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

Mapper3DNode::Mapper3DNode(const Mapper3DNodeParams &params)
    : Node("mapper3d_node"),
      mapper3d_(params.mapper3d_params),
      map_frame_id_(params.map_frame_id),
      odom_frame_id_(params.odom_frame_id),
      base_frame_id_(params.base_frame_id),
      sensor_frame_id_(params.sensor_frame_id) {
    RCLCPP_INFO(this->get_logger(), "Initializing Mapper3DNode...");

    point_cloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        params.point_cloud_sub_topic, params.sub_queue_size,
        std::bind(&Mapper3DNode::pointcloud_callback, this, std::placeholders::_1));

    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(this);

    RCLCPP_INFO(this->get_logger(), "Mapper3DNode initialized.");
}

void Mapper3DNode::pointcloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(*msg, *cloud);

    mapper3d_.spin_once(cloud);
    const Sophus::SE3d map_to_sensor_transform = mapper3d_.sensor_pose();

    broadcast_transform(map_to_sensor_transform, map_frame_id_, sensor_frame_id_, msg->header.stamp);

    std::cout << "--------------------------------------------------------------------------" << std::endl;
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

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);

    Mapper3DNodeParams params;
    auto node = std::make_shared<Mapper3DNode>(params);

    rclcpp::spin(node);
    rclcpp::shutdown();

    return 0;
}