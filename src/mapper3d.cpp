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
#include "mapper3d/graph_optimizer.h"
#include "mapper3d/utils.h"

namespace mapper3d {

Mapper3D::Mapper3D(const Mapper3DParams& params)
    : prev_point_cloud_(new pcl::PointCloud<pcl::PointXYZ>),
      point_cloud_(new pcl::PointCloud<pcl::PointXYZ>),
      prev_sensor_pose_(Sophus::SE3d()),
      sensor_pose_(Sophus::SE3d()),
      z_height_threshold_(params.z_height_threshold),
      min_sensor_distance_(params.min_sensor_distance),
      kiss_resolution_(params.voxel_size),
      kiss_config_(kiss_resolution_),
      kiss_matcher_(kiss_config_),
      graph_optimizer_(params.graph_optimizer_params),
      constrain_pose_to_se2_(false),
      viewer_initialized_(false),
      max_point_cloud_history_(50),
      pose_id_(0) {
    // stat_outlier_removal_.setMeanK(50);             // Number of neighbors to analyze for each point
    // stat_outlier_removal_.setStddevMulThresh(1.0);  // Standard deviation multiplier
    // radius_outlier_removal_.setRadiusSearch(0.8);        // Radius in meters
    // radius_outlier_removal_.setMinNeighborsInRadius(2);  // Minimum number of neighbors within the radius

    voxel_grid_.setLeafSize(params.voxel_size, params.voxel_size, params.voxel_size);

    small_gicp_.setNumThreads(params.gicp_params.num_threads);
    small_gicp_.setNumNeighborsForCovariance(params.gicp_params.num_neighbors_for_covariance);
    small_gicp_.setMaxCorrespondenceDistance(params.gicp_params.max_correspondence_distance);
    small_gicp_.setVoxelResolution(params.gicp_params.voxel_size);
    small_gicp_.setRegistrationType(params.gicp_params.registration_type);

    // viz_thread_ = std::thread(&Mapper3D::visualization_thread_func, this);
    // run_viz_thread_ = true;
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

    sensor_pose_ = estimate_lidar_odometry(prev_point_cloud_, point_cloud_, prev_sensor_pose_);

    // TODO: Use a more representative pose uncertainty/covariance
    graph_optimizer_.add_frame(Frame(pose_id_, sensor_pose_, point_cloud_, Eigen::Matrix<double, 6, 6>::Identity()));

    enqueue_visualization_data(point_cloud_, sensor_pose_);

    prev_point_cloud_ = point_cloud_;
    prev_sensor_pose_ = sensor_pose_;
    pose_id_++;
}

Sophus::SE3d Mapper3D::estimate_lidar_odometry(const pcl::PointCloud<pcl::PointXYZ>::Ptr prev_point_cloud,
                                               const pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud,
                                               const Sophus::SE3d& prev_sensor_pose) {
    Timer timer("Mapper3D::estimate_lidar_odometry");

    const auto& point_cloud_vec = utils::point_cloud_to_vec(point_cloud);
    if (!prev_point_cloud_vec_.empty() && !point_cloud_vec.empty()) {
        const auto solution = kiss_matcher_.estimate(point_cloud_vec, prev_point_cloud_vec_);
        // kiss_matcher_.print();
        const Sophus::SE3d sensor_pose_change = Sophus::SE3d(solution.rotation, solution.translation);

        // const Sophus::SE3d refined_pose_change = gicp_registration(prev_point_cloud_, point_cloud_,
        // sensor_pose_change);
        const Sophus::SE3d refined_pose_change =
            small_gicp_registration(prev_point_cloud, point_cloud, sensor_pose_change);
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

    prev_point_cloud_vec_ = point_cloud_vec;

    return sensor_pose_;
}

void Mapper3D::enqueue_visualization_data(const pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud,
                                          const Sophus::SE3d& sensor_pose) {
    auto frame = std::make_shared<PointCloudFrame>();
    frame->point_cloud = point_cloud;
    frame->sensor_pose = sensor_pose;

    {
        std::scoped_lock<std::mutex> lock(viz_mutex_);
        // Prevent queue from growing too large if visualization can't keep up
        while (viz_queue_.size() >= max_point_cloud_history_) {
            viz_queue_.pop();
        }
        viz_queue_.push(frame);
    }
    viz_cv_.notify_one();
}

Sophus::SE3d Mapper3D::constrain_pose_to_se2(const Sophus::SE3d& pose) {
    Eigen::Vector3d translation = pose.translation();
    translation.z() = 0.0;

    const Eigen::Matrix3d rotation_matrix = pose.rotationMatrix();

    // Calculate yaw (rotation around Z-axis)
    // This assumes a ZYX extrinsic Euler angle convention (yaw, pitch, roll)
    // yaw = atan2(R(1,0), R(0,0))
    const double yaw = std::atan2(rotation_matrix(1, 0), rotation_matrix(0, 0));

    const Sophus::SO3d constrained_rotation = Sophus::SO3d::rotZ(yaw);

    return Sophus::SE3d(constrained_rotation, translation);
}

pcl::PointCloud<pcl::PointXYZ>::Ptr Mapper3D::filter_point_cloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud,
                                                                 const Sophus::SE3d& prev_sensor_pose,
                                                                 const Sophus::SE3d& sensor_pose) {
    Timer timer("Mapper3D::filter_point_cloud");

    std::cout << "Number of points in original point cloud: " << point_cloud->size() << std::endl;

    // Use two pointers to manage intermediate results
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in = point_cloud;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_out(new pcl::PointCloud<pcl::PointXYZ>);

    // 1. Remove NaN points
    std::vector<int> nan_indices;
    pcl::removeNaNFromPointCloud(*cloud_in, *cloud_out, nan_indices);
    cloud_in.swap(cloud_out);
    std::cout << "Points after NaN removal: " << cloud_in->size() << std::endl;

    // 2. Remove points near the sensor
    cloud_out = remove_points_near_sensor(cloud_in, min_sensor_distance_);
    cloud_in.swap(cloud_out);
    std::cout << "Points after near sensor removal: " << cloud_in->size() << std::endl;

    // 3. Remove points below z-height threshold
    cloud_out = remove_points_below_z(cloud_in, z_height_threshold_);
    cloud_in.swap(cloud_out);
    std::cout << "Points after z-height removal: " << cloud_in->size() << std::endl;

    // // 4. Statistical Outlier Removal
    // stat_outlier_removal_.setInputCloud(cloud_in);
    // stat_outlier_removal_.filter(*cloud_out);
    // cloud_in.swap(cloud_out);
    // std::cout << "Points after statistical outlier removal: " << cloud_in->size() << std::endl;

    // // 5. Radius Outlier Removal
    // radius_outlier_removal_.setInputCloud(cloud_in);
    // radius_outlier_removal_.filter(*cloud_out);
    // cloud_in.swap(cloud_out);
    // std::cout << "Points after radius outlier removal: " << cloud_in->size() << std::endl;

    // 6. Downsample the remaining points
    voxel_grid_.setInputCloud(cloud_in);
    voxel_grid_.filter(*cloud_out);
    std::cout << "Number of points after downsampling: " << cloud_out->size() << std::endl;
    // The final result is now in cloud_out

    return cloud_out;  // Return the final filtered cloud
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

Sophus::SE3d Mapper3D::small_gicp_registration(const pcl::PointCloud<pcl::PointXYZ>::Ptr target,
                                               const pcl::PointCloud<pcl::PointXYZ>::Ptr source,
                                               const Sophus::SE3d& initial_transform_guess) {
    Timer timer("Mapper3D::small_gicp_registration");

    small_gicp_.setInputTarget(target);
    small_gicp_.setInputSource(source);

    pcl::PointCloud<pcl::PointXYZ> aligned_point_cloud;
    small_gicp_.align(aligned_point_cloud, initial_transform_guess.matrix().cast<float>());

    return eigen_to_sophus(small_gicp_.getFinalTransformation().cast<double>());
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
        std::shared_ptr<PointCloudFrame> frame;

        {
            std::unique_lock<std::mutex> lock(viz_mutex_);
            viz_cv_.wait(lock, [this]() { return !viz_queue_.empty() || !run_viz_thread_; });

            if (!run_viz_thread_) {
                break;
            }

            if (!viz_queue_.empty()) {
                frame = viz_queue_.front();
                viz_queue_.pop();
            }
        }

        if (frame) {
            visualize_aligned_point_clouds(frame->point_cloud, frame->sensor_pose, 1920, 1080, 100.0);
        }
    }
}

void Mapper3D::visualize_aligned_point_clouds(const pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud, Sophus::SE3d& pose,
                                              const size_t h_res, const size_t v_res, const double view_height) {
    Timer timer("Mapper3D::visualize_aligned_point_clouds");

    // Initialize viewer if needed
    if (!viewer_initialized_) {
        point_cloud_ids_.clear();
        point_cloud_counter_ = 0;

        viewer_ = pcl::visualization::PCLVisualizer::Ptr(new pcl::visualization::PCLVisualizer("3D Viewer"));
        viewer_->setBackgroundColor(0, 0, 0);
        viewer_->initCameraParameters();
        viewer_->setSize(h_res, v_res);
        viewer_initialized_ = true;
    }

    // If we've reached the maximum number of point clouds, remove the oldest one
    if (point_cloud_ids_.size() >= max_point_cloud_history_ && !point_cloud_ids_.empty()) {
        std::string oldest_id = point_cloud_ids_.front();
        viewer_->removePointCloud(oldest_id);
        point_cloud_ids_.pop_front();
    }

    // Create a unique ID for current point cloud
    const std::string point_cloud_id = "point_cloud_" + std::to_string(point_cloud_counter_++);
    point_cloud_ids_.push_back(point_cloud_id);

    // Transform current point cloud from sensor frame to world frame
    pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_point_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    const Eigen::Matrix4f transform = pose.matrix().cast<float>();
    pcl::transformPointCloud(*point_cloud, *transformed_point_cloud, transform);

    // Add current point cloud to viewer
    const pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler(transformed_point_cloud, 255,
                                                                                        255, 255);
    viewer_->addPointCloud(transformed_point_cloud, color_handler, point_cloud_id);
    viewer_->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, point_cloud_id);

    // Update or add coordinate system
    viewer_->removeAllCoordinateSystems();
    const Eigen::Affine3f current_transform(pose.matrix().cast<float>());
    viewer_->addCoordinateSystem(1.0, current_transform);

    // Set viewer camera position
    const Eigen::Vector3d position = pose.translation();
    viewer_->setCameraPosition(position.x(), position.y(), position.z() + view_height,  // Camera position
                               position.x(), position.y(), position.z(),                // Look at point
                               0.0, 1.0, 0.0);                                          // Up vector

    viewer_->spinOnce();
}

}  // namespace mapper3d