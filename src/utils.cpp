#include "mapper3d/utils.h"

#include "mapper3d/timer.h"

namespace mapper3d {
namespace utils {

std::vector<Eigen::Vector3f> point_cloud_to_vec(const pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud) {
    Timer timer("Mapper3D::point_cloud_to_vec");

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
    Timer timer("Mapper3D::point_vec_to_cloud");

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

double deg2rad(const double deg) { return deg * (M_PI / 180.0); }

double rad2deg(const double rad) { return rad * (180.0 / M_PI); }

}  // namespace utils
}  // namespace mapper3d