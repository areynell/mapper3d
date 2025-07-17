#include "mapper3d/utils.h"

#include <vector>

#include "mapper3d/logging.h"

namespace mapper3d {
namespace utils {

std::vector<Eigen::Vector3f> point_cloud_to_vec(const pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud) {
    Timer timer(__func__, LogLevel::Debug);

    std::vector<Eigen::Vector3f> point_vec;
    point_vec.reserve(point_cloud->size());

    for (const auto& point : point_cloud->points) {
        if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z)) {
            continue;
        }
        point_vec.emplace_back(point.x, point.y, point.z);
    }
    return point_vec;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr point_vec_to_cloud(const std::vector<Eigen::Vector3f>& point_vec) {
    Timer timer(__func__, LogLevel::Debug);

    pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    point_cloud->reserve(point_vec.size());

    for (const auto& point : point_vec) {
        if (!std::isfinite(point.x()) || !std::isfinite(point.y()) || !std::isfinite(point.z())) {
            continue;
        }
        point_cloud->emplace_back(point.x(), point.y(), point.z());
    }

    point_cloud->width = point_cloud->size();
    point_cloud->height = 1;       // Set height to 1 for unorganized point cloud
    point_cloud->is_dense = true;  // All points are finite

    return point_cloud;
}

Sophus::SE3d small_gicp_registration(const std::vector<Eigen::Vector3f>& target_points,
                                     const std::vector<Eigen::Vector3f>& source_points,
                                     const Sophus::SE3d& initial_transform_guess) {
    Timer timer(__func__, LogLevel::Debug);

    // const Eigen::Isometry3d init_guess(initial_transform_guess.matrix());

    small_gicp::RegistrationSetting setting;
    // small_gicp::RegistrationResult result = small_gicp::align(target_points, source_points, init_guess, setting);
    small_gicp::RegistrationResult result =
        small_gicp::align(target_points, source_points, Eigen::Isometry3d(initial_transform_guess.matrix()), setting);

    Logger::debug("Small GICP num inliers: " + std::to_string(result.num_inliers));
    Logger::debug("Small GICP error: " + std::to_string(result.error));
    Logger::debug("Small GICP converged: " + std::to_string(result.converged));

    return eigen_to_sophus(result.T_target_source.matrix());
}

// Sophus::SE3d eigen_to_sophus(const Eigen::Matrix4d& eigen_matrix) {
//     // Sophus requires a valid rotation matrix in SO(3):
//     // 1) The matrix must be orthogonal (R * R^T = I)
//     // 2) The determinant must be 1

//     // Extract rotation matrix
//     Eigen::Matrix3d rotation = eigen_matrix.block<3, 3>(0, 0);

//     // Ensure rotation is orthogonal using SVD
//     Eigen::JacobiSVD<Eigen::Matrix3d> svd(rotation, Eigen::ComputeFullU | Eigen::ComputeFullV);
//     Eigen::Matrix3d orthogonal_rotation = svd.matrixU() * svd.matrixV().transpose();

//     // Ensure determinant is 1
//     if (orthogonal_rotation.determinant() < 0) {
//         Eigen::Matrix3d correction;
//         correction << 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0;
//         orthogonal_rotation = orthogonal_rotation * correction;
//     }

//     // Extract translation
//     Eigen::Vector3d translation = eigen_matrix.block<3, 1>(0, 3);

//     return Sophus::SE3d(orthogonal_rotation, translation);
// }

Sophus::SE3d eigen_to_sophus(const Eigen::Matrix4d& eigen_matrix) {
    // Extract rotation matrix
    Eigen::Matrix3d rotation = eigen_matrix.block<3, 3>(0, 0);

    // Check if the rotation matrix is already close to orthogonal
    double orthogonality_error = (rotation * rotation.transpose() - Eigen::Matrix3d::Identity()).norm();

    if (orthogonality_error > 1.0e-6) {
        // Use SVD to find the closest orthogonal matrix
        Eigen::JacobiSVD<Eigen::Matrix3d> svd(rotation, Eigen::ComputeFullU | Eigen::ComputeFullV);
        rotation = svd.matrixU() * svd.matrixV().transpose();

        // If determinant is negative, we have a reflection - fix by negating the last column of V
        if (rotation.determinant() < 0) {
            Eigen::Matrix3d V = svd.matrixV();
            V.col(2) *= -1.0;  // Flip the last column
            rotation = svd.matrixU() * V.transpose();
        }
    }

    Eigen::Vector3d translation = eigen_matrix.block<3, 1>(0, 3);

    return Sophus::SE3d(rotation, translation);
}

Sophus::SE3d constrain_pose_to_se2(const Sophus::SE3d& pose) {
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

double deg2rad(const double deg) { return deg * (M_PI / 180.0); }

double rad2deg(const double rad) { return rad * (180.0 / M_PI); }

}  // namespace utils
}  // namespace mapper3d