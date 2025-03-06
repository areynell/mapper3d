#include "mapper3d/mapper3d.h"

#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <limits>

#include "mapper3d/common.h"

std::vector<Eigen::Vector3f> Mapper3D::convert_point_cloud_to_vec(const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud) {
    std::vector<Eigen::Vector3f> cloud_vec;
    cloud_vec.reserve(cloud->size());

    for (const auto& point : cloud->points) {
        if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z)) {
            continue;
        }
        cloud_vec.emplace_back(point.x, point.y, point.z);
    }
    return cloud_vec;
}

Mapper3D::Mapper3D(const Mapper3DParams& params)
    : prev_point_cloud_(new pcl::PointCloud<pcl::PointXYZ>),
      prev_sensor_pose_(Sophus::SE3d()),
      sensor_pose_(Sophus::SE3d()),
      z_height_threshold_(-1.0),
      kiss_resolution_(0.2),
      kiss_config_(kiss_resolution_),
      kiss_matcher_(kiss_config_),
      viewer_initialized_(false) {
    // icp_.setMaximumIterations(params.max_iterations);
    // icp_.setMaxCorrespondenceDistance(params.max_correspondence_distance);
    // icp_.setTransformationEpsilon(params.transformation_epsilon);
    // icp_.setEuclideanFitnessEpsilon(params.euclidean_fitness_epsilon);
    // icp_.setRANSACOutlierRejectionThreshold(params.ransac_outlier_rejection_threshold);
    // icp_.setRANSACIterations(params.ransac_iterations);

    voxel_grid_.setLeafSize(params.voxel_size, params.voxel_size, params.voxel_size);
}

void Mapper3D::spin_once(const pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud) {
    Timer timer("Mapper3D::spin_once");

    if (point_cloud->empty()) {
        return;
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr downsampled_point_cloud = downsample_point_cloud(point_cloud);

    std::vector<int> nan_indices;
    pcl::removeNaNFromPointCloud(*downsampled_point_cloud, *downsampled_point_cloud, nan_indices);

    // downsampled_point_cloud = remove_ground_plane(downsampled_point_cloud);
    downsampled_point_cloud = remove_points_below_z(downsampled_point_cloud, z_height_threshold_);

    if (prev_point_cloud_->empty()) {
        *prev_point_cloud_ = *downsampled_point_cloud;
        return;
    }

    const auto& point_cloud_vec = convert_point_cloud_to_vec(downsampled_point_cloud);

    if (!prev_point_cloud_vec_.empty() && !point_cloud_vec.empty()) {
        const auto solution = kiss_matcher_.estimate(point_cloud_vec, prev_point_cloud_vec_);
        const Sophus::SE3d sensor_pose_change = Sophus::SE3d(solution.rotation, solution.translation);
        sensor_pose_ = prev_sensor_pose_ * sensor_pose_change;
        kiss_matcher_.print();
    }

    // visualize_point_cloud(downsampled_point_cloud);

    *prev_point_cloud_ = *downsampled_point_cloud;
    prev_point_cloud_vec_ = point_cloud_vec;
    prev_sensor_pose_ = sensor_pose_;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr Mapper3D::downsample_point_cloud(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud) {
    auto filtered_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();

    voxel_grid_.setInputCloud(input_cloud);
    voxel_grid_.filter(*filtered_cloud);
    return filtered_cloud;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr Mapper3D::remove_ground_plane(const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud) {
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_no_ground(new pcl::PointCloud<pcl::PointXYZ>);

    // Create segmentation object
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PERPENDICULAR_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(0.2);  // 20cm threshold
    seg.setMaxIterations(100);

    // Assume ground is perpendicular to Z axis
    Eigen::Vector3f axis = Eigen::Vector3f(0.0, 0.0, 1.0);
    seg.setAxis(axis);
    seg.setEpsAngle(15.0 * (M_PI / 180.0));  // 15 degree tolerance

    seg.setInputCloud(cloud);
    seg.segment(*inliers, *coefficients);

    // Extract non-ground points
    pcl::ExtractIndices<pcl::PointXYZ> extract;
    extract.setInputCloud(cloud);
    extract.setIndices(inliers);
    extract.setNegative(true);  // Keep everything except the ground
    extract.filter(*cloud_no_ground);

    return cloud_no_ground;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr Mapper3D::remove_points_below_z(const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
                                                                    const double z_threshold) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);

    // Create the passthrough filter
    pcl::PassThrough<pcl::PointXYZ> pass;
    pass.setInputCloud(cloud);
    pass.setFilterFieldName("z");                                          // Filter based on z-coordinate
    pass.setFilterLimits(z_threshold, std::numeric_limits<float>::max());  // Keep points with z > z_threshold

    // Apply the filter
    pass.filter(*cloud_filtered);

    return cloud_filtered;
}

Sophus::SE3d Mapper3D::gicp_registration(const pcl::PointCloud<pcl::PointXYZ>::Ptr source,
                                         const pcl::PointCloud<pcl::PointXYZ>::Ptr target) {
    Timer timer("Mapper3D::gicp_registration");

    gicp_.setInputSource(source);
    gicp_.setInputTarget(target);

    pcl::PointCloud<pcl::PointXYZ> aligned_point_cloud;
    gicp_.align(aligned_point_cloud);

    if (gicp_.hasConverged()) {
        std::cout << "ICP converged." << std::endl;
    } else {
        std::cout << "ICP did not converge." << std::endl;
    }

    std::cout << "GICP fitness score: " << gicp_.getFitnessScore() << std::endl;
    std::cout << "GICP transformation matrix: " << std::endl;
    std::cout << gicp_.getFinalTransformation() << std::endl;

    return eigen_to_sophus(gicp_.getFinalTransformation().cast<double>());
}

// pcl::PointCloud<PointXYZT>::Ptr Mapper3D::dewarp_point_cloud(const pcl::PointCloud<PointXYZT>::Ptr &point_cloud,
//                                                              const Eigen::Matrix4f &prev_sensor_pose,
//                                                              const Eigen::Matrix4f &sensor_pose) {
//     Timer timer("Mapper3D::dewarp_point_cloud");
// }

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

Sophus::SE3d Mapper3D::eigen_to_sophus(const Eigen::Matrix4d& eigen_matrix) {
    return Sophus::SE3d(eigen_matrix.block<3, 3>(0, 0), eigen_matrix.block<3, 1>(0, 3));
}

void Mapper3D::visualize_point_cloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud) {
    // Initialize viewer if needed
    if (!viewer_initialized_) {
        viewer_ = pcl::visualization::PCLVisualizer::Ptr(new pcl::visualization::PCLVisualizer("3D Viewer"));
        viewer_->setBackgroundColor(0, 0, 0);
        viewer_->addCoordinateSystem(1.0);
        viewer_->initCameraParameters();

        // Set camera position for top-down view
        // Position the camera high above the z-axis looking down
        viewer_->setCameraPosition(0.0, 0.0, 25.0,  // Camera position: high above on z-axis
                                   0.0, 0.0, 0.0,   // Looking at origin
                                   0.0, 1.0, 0.0    // Up vector (y-axis is "up" in top-down view)
        );

        viewer_initialized_ = true;
    }

    // Add the new point cloud
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler(point_cloud, 255, 255, 255);
    viewer_->addPointCloud(point_cloud, color_handler, "cloud");
    viewer_->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "cloud");

    // SPINONCE IS CAUSING SEGFAULT
    viewer_->spinOnce();

    // Remove all existing point clouds before adding new ones
    viewer_->removeAllPointClouds();
}

void Mapper3D::add_keypoints_to_visualizer(const pcl::PointCloud<pcl::PointXYZ>::Ptr keypoints) {
    if (!viewer_initialized_) {
        std::cerr << "Viewer not initialized. Call visualize_point_cloud() first." << std::endl;
        return;
    }

    // Add the keypoints to the viewer
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler(keypoints, 255, 0, 0);
    viewer_->addPointCloud(keypoints, color_handler, "keypoints");
    viewer_->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "keypoints");
}
