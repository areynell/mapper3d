#include "mapper3d/mapper3d.h"

#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <cmath>
#include <limits>
#include <mutex>
#include <thread>
#include <vector>

#include "mapper3d/common.h"
#include "mapper3d/utils.h"

namespace mapper3d {

Mapper3D::Mapper3D(const Mapper3DParams& params)
    : prev_point_cloud_(new pcl::PointCloud<pcl::PointXYZ>),
      point_cloud_(new pcl::PointCloud<pcl::PointXYZ>),
      prev_sensor_pose_(Sophus::SE3d()),
      sensor_pose_(Sophus::SE3d()),
      z_height_threshold_(-1.0),
      kiss_resolution_(params.voxel_size),
      kiss_config_(kiss_resolution_),
      kiss_matcher_(kiss_config_),
      constrain_pose_to_se2_(false),
      viewer_initialized_(false),
      max_point_cloud_history_(50),
      new_viz_data_available_(false) {
    std::cout << "GICP maximum correspondence distance: " << gicp_.getMaxCorrespondenceDistance() << "\n";
    // gicp_.setMaxCorrespondenceDistance(0.1);

    voxel_grid_.setLeafSize(params.voxel_size, params.voxel_size, params.voxel_size);

    viz_thread_ = std::thread(&Mapper3D::visualization_thread_func, this);
    run_viz_thread_ = true;
}

// Mapper3D::~Mapper3D() {
//     {
//         std::unique_lock<std::mutex> lock(viz_mutex_);
//         run_viz_thread_ = false;
//         new_viz_data_available_ = true;  // Ensure the thread wakes up
//     }
//     viz_cv_.notify_one();
//     if (viz_thread_.joinable()) {
//         viz_thread_.join();
//     }
// }

void Mapper3D::spin_once(const pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud) {
    Timer timer("Mapper3D::spin_once");

    if (point_cloud->empty()) {
        return;
    }

    // point_cloud_ = point_cloud;
    point_cloud_ = filter_point_cloud(point_cloud, prev_sensor_pose_, sensor_pose_);

    if (prev_point_cloud_->empty()) {
        prev_point_cloud_ = point_cloud_;
        return;
    }

    const auto& point_cloud_vec = utils::point_cloud_to_vec(point_cloud_);
    if (!prev_point_cloud_vec_.empty() && !point_cloud_vec.empty()) {
        const auto solution = kiss_matcher_.estimate(point_cloud_vec, prev_point_cloud_vec_);
        // kiss_matcher_.print();
        const Sophus::SE3d sensor_pose_change = Sophus::SE3d(solution.rotation, solution.translation);
        const Sophus::SE3d refined_pose_change = gicp_registration(prev_point_cloud_, point_cloud_, sensor_pose_change);
        sensor_pose_ = prev_sensor_pose_ * refined_pose_change;

        if (constrain_pose_to_se2_) {
            sensor_pose_ = constrain_pose_to_se2(sensor_pose_);
        }
    }

    const auto kiss_matcher_processed_point_cloud_pair = kiss_matcher_.getProcessedInputClouds();
    kiss_matcher_processed_prev_point_cloud_ = std::get<0>(kiss_matcher_processed_point_cloud_pair);
    kiss_matcher_processed_point_cloud_ = std::get<1>(kiss_matcher_processed_point_cloud_pair);

    const auto kiss_matcher_keypoint_cloud_pair = kiss_matcher_.getKeypointsFromFasterPFH();
    kiss_matcher_prev_keypoints_ = std::get<0>(kiss_matcher_keypoint_cloud_pair);
    kiss_matcher_keypoints_ = std::get<1>(kiss_matcher_keypoint_cloud_pair);

    const auto kiss_matcher_correspondences = kiss_matcher_.getFinalCorrespondences();

    {
        Timer timer("Mapper3D::spin_once locked viz_mutex_ for");

        PointCloudFrame frame;
        frame.cloud = point_cloud_;
        frame.pose = sensor_pose_;

        std::unique_lock<std::mutex> lock(viz_mutex_);
        aligned_point_cloud_frames_.push_back(frame);
        if (aligned_point_cloud_frames_.size() > max_point_cloud_history_) {
            aligned_point_cloud_frames_.pop_front();
        }
        new_viz_data_available_ = true;
        viz_cv_.notify_one();
    }

    prev_point_cloud_ = point_cloud_;
    prev_point_cloud_vec_ = point_cloud_vec;
    prev_sensor_pose_ = sensor_pose_;
}

Sophus::SE3d Mapper3D::constrain_pose_to_se2(const Sophus::SE3d& pose) {
    Eigen::Vector3d translation = pose.translation();
    translation.z() = 0.0;
    Eigen::Vector3d euler_angles = pose.so3().log();
    euler_angles.x() = 0.0;
    euler_angles.y() = 0.0;
    return Sophus::SE3d(Sophus::SO3d::exp(euler_angles), translation);
}

pcl::PointCloud<pcl::PointXYZ>::Ptr Mapper3D::filter_point_cloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud,
                                                                 const Sophus::SE3d& prev_sensor_pose,
                                                                 const Sophus::SE3d& sensor_pose) {
    std::cout << "Number of points in original point cloud: " << point_cloud->size() << std::endl;

    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_point_cloud = downsample_point_cloud(point_cloud);
    filtered_point_cloud = remove_points_below_z(filtered_point_cloud, z_height_threshold_);
    filtered_point_cloud = remove_points_near_sensor(filtered_point_cloud, 3.0);

    std::vector<int> nan_indices;
    pcl::removeNaNFromPointCloud(*filtered_point_cloud, *filtered_point_cloud, nan_indices);

    std::cout << "Number of points in filtered point cloud: " << filtered_point_cloud->size() << std::endl;

    return filtered_point_cloud;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr Mapper3D::downsample_point_cloud(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud) {
    Timer timer("Mapper3D::downsample_point_cloud");

    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    voxel_grid_.setInputCloud(input_cloud);
    voxel_grid_.filter(*filtered_cloud);

    return filtered_cloud;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr Mapper3D::remove_points_below_z(const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
                                                                    const double z_threshold) {
    Timer timer("Mapper3D::remove_points_below_z");

    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>);

    pcl::PassThrough<pcl::PointXYZ> passthrough;
    passthrough.setInputCloud(cloud);
    passthrough.setFilterFieldName("z");                                          // Filter based on z-coordinate
    passthrough.setFilterLimits(z_threshold, std::numeric_limits<float>::max());  // Keep points with z > z_threshold
    passthrough.filter(*filtered_cloud);

    return filtered_cloud;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr Mapper3D::remove_points_near_sensor(const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
                                                                        const double min_distance) {
    Timer timer("Mapper3D::remove_points_near_sensor");

    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    filtered_cloud->reserve(cloud->size());

    for (const auto& point : cloud->points) {
        const double distance_squared = point.x * point.x + point.y * point.y + point.z * point.z;
        if (distance_squared > min_distance * min_distance) {
            filtered_cloud->points.push_back(point);
        }
    }

    filtered_cloud->width = filtered_cloud->points.size();
    filtered_cloud->height = 1;
    filtered_cloud->is_dense = false;

    return filtered_cloud;
}

Sophus::SE3d Mapper3D::gicp_registration(const pcl::PointCloud<pcl::PointXYZ>::Ptr target,
                                         const pcl::PointCloud<pcl::PointXYZ>::Ptr source,
                                         const Sophus::SE3d& initial_transform_guess) {
    Timer timer("Mapper3D::gicp_registration");

    gicp_.setInputTarget(target);
    gicp_.setInputSource(source);

    pcl::PointCloud<pcl::PointXYZ> aligned_point_cloud;
    gicp_.align(aligned_point_cloud, initial_transform_guess.matrix().cast<float>());

    if (!gicp_.hasConverged()) {
        std::cout << "Warning: GICP did not converge." << std::endl;
    }

    std::cout << "GICP fitness score: " << gicp_.getFitnessScore() << std::endl;
    // std::cout << "GICP transformation matrix: " << std::endl;
    // std::cout << gicp_.getFinalTransformation() << std::endl;

    return eigen_to_sophus(gicp_.getFinalTransformation().cast<double>());
}

// Sophus::SE3d Mapper3D::predict_sensor_motion() {
//     // IMPORTANT: Assumes that dt between laser scans is constant
//     // If the above assumption is true, we can use the pose change between pose(t-1) and pose(t) to predict
//     pose(t+1)

//     // Since:
//     // twist = log(pose_change)/dt
//     // and:
//     // predicted_pose = prev_pose * exp(twist * dt)
//     // we can write:
//     // predicted_pose = prev_pose * exp(log(pose_change)/dt * dt)
//     // predicted_pose = prev_pose * exp(log(pose_change))
//     // predicted_pose = prev_pose * pose_change

//     if (sensor_pose_history_.size() < 2) {
//         return Sophus::SE3d(Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero());
//     }

//     const Sophus::SE3d sensor_pose_tm2 = sensor_pose_history_[sensor_pose_history_.size() - 2];
//     const Sophus::SE3d sensor_pose_tm1 = sensor_pose_history_[sensor_pose_history_.size() - 1];
//     const Sophus::SE3d sensor_pose_change = sensor_pose_tm2.inverse() * sensor_pose_tm1;

//     return sensor_pose_change;
// }

Sophus::SE3d Mapper3D::sensor_pose() const { return sensor_pose_; }

pcl::PointCloud<pcl::PointXYZ>::Ptr Mapper3D::point_cloud() const { return point_cloud_; }

const std::vector<Eigen::Vector3f>& Mapper3D::kiss_matcher_point_cloud() const {
    return kiss_matcher_processed_point_cloud_;
}

const std::vector<Eigen::Vector3f>& Mapper3D::kiss_matcher_keypoint_cloud() const { return kiss_matcher_keypoints_; }

Sophus::SE3d Mapper3D::eigen_to_sophus(const Eigen::Matrix4d& eigen_matrix) {
    // Sophus requires a valid rotation matrix in SO(3):
    // 1) The matrix must be orthogonal (R * R^T = I)
    // 2) The determinant must be 1

    // Extract rotation matrix
    Eigen::Matrix3d rotation = eigen_matrix.block<3, 3>(0, 0);

    // Ensure rotation is orthogonal using SVD
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(rotation, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d orthogonal_rotation = svd.matrixU() * svd.matrixV().transpose();

    // Ensure determinant is 1
    if (orthogonal_rotation.determinant() < 0) {
        Eigen::Matrix3d correction;
        correction << 1, 0, 0, 0, 1, 0, 0, 0, -1;
        orthogonal_rotation = orthogonal_rotation * correction;
    }

    // Extract translation
    Eigen::Vector3d translation = eigen_matrix.block<3, 1>(0, 3);

    return Sophus::SE3d(orthogonal_rotation, translation);
}

void Mapper3D::visualization_thread_func() {
    while (run_viz_thread_) {
        // Timer timer("Mapper3D::visualization_thread_func loop time");

        std::deque<PointCloudFrame> aligned_point_cloud_frames_copy;

        {
            // Timer timer2("Mapper3D::visualization_thread_func locked viz_mutex_ for");

            std::unique_lock<std::mutex> lock(viz_mutex_);
            viz_cv_.wait(lock, [this]() { return new_viz_data_available_ || !run_viz_thread_; });

            if (!run_viz_thread_) {
                break;
            }

            if (new_viz_data_available_ && !aligned_point_cloud_frames_.empty()) {
                Timer timer3("Mapper3D::visualization_thread_func copy aligned_point_cloud_frames_");
                // Make a copy of the data we need while holding the lock
                aligned_point_cloud_frames_copy = aligned_point_cloud_frames_;
                new_viz_data_available_ = false;
            }
        }

        // Visualize using local copy without holding the lock
        if (!aligned_point_cloud_frames_copy.empty()) {
            visualize_aligned_point_clouds(aligned_point_cloud_frames_copy, 1920, 1080, 100.0);
        }
    }
}

void Mapper3D::visualize_aligned_point_clouds(const std::deque<PointCloudFrame>& aligned_point_cloud_frames,
                                              const size_t h_res, const size_t v_res, const double view_height) {
    Timer timer("Mapper3D::visualize_aligned_point_clouds");

    // Initialize viewer if needed
    if (!viewer_initialized_) {
        viewer_ = pcl::visualization::PCLVisualizer::Ptr(new pcl::visualization::PCLVisualizer("3D Viewer"));
        viewer_->setBackgroundColor(0, 0, 0);
        viewer_->initCameraParameters();
        viewer_->setSize(h_res, v_res);
        viewer_initialized_ = true;
    }

    // Remove existing point clouds and coordinate axes
    viewer_->removeAllPointClouds();
    viewer_->removeAllCoordinateSystems();

    // Add each point cloud to viewer
    size_t idx = 0;
    for (const auto& frame : aligned_point_cloud_frames) {
        pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        Eigen::Matrix4f transform = frame.pose.matrix().cast<float>();
        pcl::transformPointCloud(*frame.cloud, *transformed_cloud, transform);

        std::string cloud_id = "cloud_" + std::to_string(idx);
        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler(transformed_cloud, 255, 255, 255);
        viewer_->addPointCloud(transformed_cloud, color_handler, cloud_id);
        viewer_->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, cloud_id);
        idx++;
    }

    if (!aligned_point_cloud_frames.empty()) {
        // Add coordinate axis at the current sensor pose
        Eigen::Affine3f current_transform(aligned_point_cloud_frames.back().pose.matrix().cast<float>());
        viewer_->addCoordinateSystem(1.0, current_transform);

        // Set camera position
        Eigen::Vector3d position = aligned_point_cloud_frames.back().pose.translation();
        viewer_->setCameraPosition(position.x(), position.y(), position.z() + view_height,  // Camera position
                                   position.x(), position.y(), position.z(),                // Look at point
                                   0.0, 1.0, 0.0);                                          // Up vector
    }

    viewer_->spinOnce();
}

}  // namespace mapper3d