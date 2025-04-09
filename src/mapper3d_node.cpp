#include "mapper3d/mapper3d_node.h"

#include <pcl_conversions/pcl_conversions.h>
#include <tf2/convert.h>

#include <geometry_msgs/msg/transform_stamped.hpp>
#include <mutex>
#include <rclcpp/logging.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include "mapper3d/common.h"
#include "mapper3d/utils.h"

Mapper3DNode::Mapper3DNode(const Mapper3DNodeParams &params)
    : Node("mapper3d_node"),
      mapper3d_(params.mapper3d_params),
      map_frame_id_(params.map_frame_id),
      odom_frame_id_(params.odom_frame_id),
      base_frame_id_(params.base_frame_id),
      sensor_frame_id_(params.sensor_frame_id),
      run_viz_thread_(true),
      map_to_sensor_transform_(Sophus::SE3d()) {
    RCLCPP_INFO(this->get_logger(), "Initializing Mapper3DNode...");

    point_cloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        params.point_cloud_sub_topic, params.sub_queue_size,
        std::bind(&Mapper3DNode::pointcloud_callback, this, std::placeholders::_1));

    point_cloud_pub_ =
        this->create_publisher<sensor_msgs::msg::PointCloud2>(params.point_cloud_pub_topic, params.pub_queue_size);

    keypoint_pub_ =
        this->create_publisher<sensor_msgs::msg::PointCloud2>(params.keypoint_pub_topic, params.pub_queue_size);

    pose_trajectory_pub_ =
        this->create_publisher<nav_msgs::msg::Path>(params.pose_trajectory_topic, params.pub_queue_size);

    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(this);

    viz_thread_ = std::thread(&Mapper3DNode::visualization_thread_func, this);

    RCLCPP_INFO(this->get_logger(), "Mapper3DNode initialized.");
}

void Mapper3DNode::pointcloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    Timer timer("Mapper3DNode::pointcloud_callback");

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(*msg, *cloud);

    mapper3d_.spin_once(cloud);
    map_to_sensor_transform_ = mapper3d_.sensor_pose();

    broadcast_transform(map_to_sensor_transform_, map_frame_id_, sensor_frame_id_, msg->header.stamp);

    // TODO: Improve this method for copying data to the visualization thread
    {
        std::scoped_lock<std::mutex> lock(viz_mutex_);
        viz_point_cloud_ = mapper3d_.point_cloud();
        viz_kiss_matcher_point_cloud_ = mapper3d_.kiss_matcher_point_cloud();
        viz_kiss_matcher_keypoint_cloud_ = mapper3d_.kiss_matcher_keypoint_cloud();
        viz_sensor_pose_ = mapper3d_.sensor_pose();
        viz_timestamp_ = msg->header.stamp;
        new_viz_data_available_ = true;
    }
    viz_cv_.notify_one();

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

void Mapper3DNode::visualization_thread_func() {
    pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud_to_publish;
    pcl::PointCloud<pcl::PointXYZ>::Ptr keypoint_cloud_to_publish;
    Sophus::SE3d sensor_pose_to_publish;
    std::string frame_id;
    rclcpp::Time timestamp;

    while (run_viz_thread_) {
        {
            std::unique_lock<std::mutex> lock(viz_mutex_);
            viz_cv_.wait(lock, [this]() { return new_viz_data_available_ || !run_viz_thread_; });
            if (!run_viz_thread_) {
                break;
            }

            // point_cloud_to_publish = viz_point_cloud_;
            point_cloud_to_publish = mapper3d::utils::point_vec_to_cloud(viz_kiss_matcher_point_cloud_);
            keypoint_cloud_to_publish = mapper3d::utils::point_vec_to_cloud(viz_kiss_matcher_keypoint_cloud_);
            sensor_pose_to_publish = viz_sensor_pose_;
            timestamp = viz_timestamp_;
            new_viz_data_available_ = false;
        }

        publish_point_cloud(point_cloud_to_publish, sensor_frame_id_, timestamp);
        publish_keypoint_cloud(keypoint_cloud_to_publish, sensor_frame_id_, timestamp);
        publish_pose_trajectory(sensor_pose_to_publish, map_frame_id_, timestamp);
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

void Mapper3DNode::publish_pose_trajectory(const Sophus::SE3d &pose, const std::string &frame_id,
                                           const rclcpp::Time &timestamp) {
    geometry_msgs::msg::PoseStamped pose_stamped;
    pose_stamped.header.stamp = timestamp;
    pose_stamped.header.frame_id = frame_id;

    pose_stamped.pose.position.x = pose.translation().x();
    pose_stamped.pose.position.y = pose.translation().y();
    pose_stamped.pose.position.z = pose.translation().z();
    pose_stamped.pose.orientation.x = pose.unit_quaternion().x();
    pose_stamped.pose.orientation.y = pose.unit_quaternion().y();
    pose_stamped.pose.orientation.z = pose.unit_quaternion().z();
    pose_stamped.pose.orientation.w = pose.unit_quaternion().w();

    pose_trajectory_.header.stamp = timestamp;
    pose_trajectory_.header.frame_id = frame_id;
    pose_trajectory_.poses.push_back(pose_stamped);

    pose_trajectory_pub_->publish(pose_trajectory_);
}

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);

    Mapper3DNodeParams params;
    auto node = std::make_shared<Mapper3DNode>(params);

    rclcpp::spin(node);
    rclcpp::shutdown();

    return 0;
}