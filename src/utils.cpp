#include "mapper3d/utils.h"

namespace mapper3d {
namespace utils {

std::vector<Eigen::Vector3f> point_cloud_to_vec(const pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud) {
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
    pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    point_cloud->reserve(point_vec.size());

    for (const auto& point : point_vec) {
        if (!std::isfinite(point.x()) || !std::isfinite(point.y()) || !std::isfinite(point.z())) {
            continue;
        }
        point_cloud->emplace_back(point.x(), point.y(), point.z());
    }
    return point_cloud;
}

}  // namespace utils
}  // namespace mapper3d