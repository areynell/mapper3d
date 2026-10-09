#include "mapper3d/utils.h"

#include <iomanip>
#include <limits>
#include <string>
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

Eigen::Isometry3d sophus_to_isometry(const Sophus::SE3d& pose) {
    Eigen::Isometry3d isometry = Eigen::Isometry3d::Identity();
    isometry.linear() = pose.rotationMatrix();
    isometry.translation() = pose.translation();
    return isometry;
}

Sophus::SE3d eigen_to_sophus(const Eigen::Matrix4d& eigen_matrix) {
    Timer timer(__func__, LogLevel::Debug);

    // Sophus requires a valid rotation matrix in SO(3):
    // 1) The rotation matrix must be orthogonal (R * R^T = I)
    // 2) The rotation matrix determinant must be 1

    const Eigen::Matrix3d rotation = eigen_matrix.block<3, 3>(0, 0);

    // Ensure that rotation is orthogonal using SVD
    const Eigen::JacobiSVD<Eigen::Matrix3d> svd(rotation, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d orthogonal_rotation = svd.matrixU() * svd.matrixV().transpose();

    // Ensure that determinant is +1 (if not, we have a reflection, so need to flip the last column)
    if (orthogonal_rotation.determinant() < 0) {
        Logger::warn("Rotation matrix determinant is negative, flipping the last column to ensure a valid rotation.");

        Eigen::Matrix3d V = svd.matrixV();
        V.col(2) *= -1.0;  // Flip the last column
        orthogonal_rotation = svd.matrixU() * V.transpose();
    }

    const Eigen::Vector3d translation = eigen_matrix.block<3, 1>(0, 3);

    return Sophus::SE3d(orthogonal_rotation, translation);
}

double deg2rad(const double deg) { return deg * (M_PI / 180.0); }

double rad2deg(const double rad) { return rad * (180.0 / M_PI); }

double inlier_ratio(const size_t num_inliers, const size_t num_points) {
    if (num_points == 0) {
        return 0.0;
    }
    return static_cast<double>(num_inliers) / static_cast<double>(num_points);
}

double error_per_inlier(const double error, const size_t num_inliers) {
    if (num_inliers == 0) {
        return std::numeric_limits<double>::infinity();
    }
    return error / static_cast<double>(num_inliers);
}

std::ofstream open_tum_file(const std::string& path, const std::string& description) {
    std::ofstream file;

    file.open(path);
    if (!file.is_open()) {
        Logger::error("Failed to open trajectory file for writing: " + path);
        return file;
    }

    file << "# " << description << "\n";
    file << "# timestamp tx ty tz qx qy qz qw\n";

    Logger::info_important("Writing trajectory to " + path);

    return file;
}

void write_tum_pose(std::ostream& out, const double timestamp, const Sophus::SE3d& pose) {
    const Eigen::Vector3d& translation = pose.translation();
    const Eigen::Quaterniond quaternion = pose.unit_quaternion();

    out << std::fixed << std::setprecision(9) << timestamp << ' ' << translation.x() << ' ' << translation.y() << ' '
        << translation.z() << ' ' << quaternion.x() << ' ' << quaternion.y() << ' ' << quaternion.z() << ' '
        << quaternion.w() << '\n';
}

}  // namespace utils
}  // namespace mapper3d