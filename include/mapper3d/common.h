#pragma once

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <chrono>
#include <iostream>
#include <sophus/se3.hpp>
#include <string>

struct Timer {
    std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
    std::chrono::duration<float> duration;
    std::string function_name;

    Timer(const std::string &function_name) : function_name(function_name) {
        start = std::chrono::high_resolution_clock::now();
    }

    ~Timer() {
        end = std::chrono::high_resolution_clock::now();
        duration = end - start;

        const float seconds = duration.count();
        std::cout << function_name << " took " << seconds * 1000 << " ms\n";
    }
};

struct Frame {
    size_t id;
    Sophus::SE3d pose;
    pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud;
    Eigen::Matrix<double, 6, 6> odometry_covariance;

    Frame() {}

    Frame(size_t id, const Sophus::SE3d &pose, const pcl::PointCloud<pcl::PointXYZ>::Ptr &point_cloud,
          const Eigen::Matrix<double, 6, 6> &odometry_covariance)
        : id(id), pose(pose), point_cloud(point_cloud), odometry_covariance(odometry_covariance) {}
};