#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <rclcpp/rclcpp.hpp>
#include <rosbag2_cpp/writer.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/point_field.hpp>
#include <string>
#include <vector>

using namespace std::chrono_literals;

const std::string frame_id = "velodyne";
const std::string topic_name = "velodyne_points";

struct PointXYZI {
    float x, y, z, intensity;
};

// Load point cloud data from a binary file
std::vector<PointXYZI> load_velodyne_bin(const std::string &bin_path) {
    std::vector<PointXYZI> points;
    std::ifstream file(bin_path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open " + bin_path);
    }

    PointXYZI point;
    while (file.read(reinterpret_cast<char *>(&point), sizeof(PointXYZI))) {
        points.push_back(point);
    }
    return points;
}

// Load all bin files from a directory
std::vector<std::string> load_bin_files(const std::string &directory) {
    std::vector<std::string> bin_files;

    for (const auto &entry : std::filesystem::directory_iterator(directory)) {
        if (entry.path().extension() == ".bin") {
            bin_files.push_back(entry.path().string());
        }
    }

    // Sort files numerically
    std::sort(bin_files.begin(), bin_files.end(), [](const std::string &a, const std::string &b) {
        std::string base_a = std::filesystem::path(a).stem().string();
        std::string base_b = std::filesystem::path(b).stem().string();
        return std::stoul(base_a) < std::stoul(base_b);
    });

    return bin_files;
}

// Load timestamps from a text file
std::vector<double> load_timestamps(const std::string &timestamp_path) {
    std::vector<double> timestamps;
    std::ifstream file(timestamp_path);

    if (!file) {
        throw std::runtime_error("Failed to open timestamps file: " + timestamp_path);
    }

    std::string line;
    while (std::getline(file, line)) {
        // Convert string timestamp to double
        try {
            double timestamp = std::stod(line);
            timestamps.push_back(timestamp);
        } catch (const std::exception &e) {
            std::cerr << "Error parsing timestamp: " << line << " - " << e.what() << std::endl;
        }
    }

    return timestamps;
}

sensor_msgs::msg::PointCloud2 create_pointcloud_msg(const std::vector<PointXYZI> &points, double timestamp) {
    sensor_msgs::msg::PointCloud2 msg;
    msg.header.stamp.sec = static_cast<int>(timestamp);
    msg.header.stamp.nanosec = static_cast<int>((timestamp - msg.header.stamp.sec) * 1e9);
    msg.header.frame_id = frame_id;

    msg.height = 1;
    msg.width = points.size();

    msg.fields.resize(4);
    msg.fields[0].name = "x";
    msg.fields[0].offset = 0;
    msg.fields[0].datatype = sensor_msgs::msg::PointField::FLOAT32;
    msg.fields[0].count = 1;

    msg.fields[1].name = "y";
    msg.fields[1].offset = 4;
    msg.fields[1].datatype = sensor_msgs::msg::PointField::FLOAT32;
    msg.fields[1].count = 1;

    msg.fields[2].name = "z";
    msg.fields[2].offset = 8;
    msg.fields[2].datatype = sensor_msgs::msg::PointField::FLOAT32;
    msg.fields[2].count = 1;

    msg.fields[3].name = "intensity";
    msg.fields[3].offset = 12;
    msg.fields[3].datatype = sensor_msgs::msg::PointField::FLOAT32;
    msg.fields[3].count = 1;

    msg.is_bigendian = false;
    msg.point_step = sizeof(PointXYZI);
    msg.row_step = msg.point_step * points.size();
    msg.is_dense = true;

    msg.data.resize(points.size() * sizeof(PointXYZI));
    memcpy(msg.data.data(), points.data(), msg.data.size());

    return msg;
}

void save_to_rosbag(const std::vector<std::string> &bin_files, const std::vector<double> &timestamps,
                    const std::string &bag_file) {
    if (bin_files.size() != timestamps.size()) {
        throw std::runtime_error("Number of bin files (" + std::to_string(bin_files.size()) +
                                 ") does not match number of timestamps (" + std::to_string(timestamps.size()) + ")");
    }

    rosbag2_cpp::Writer writer;
    writer.open(bag_file);

    writer.create_topic({topic_name, "sensor_msgs/msg/PointCloud2", "cdr"});

    for (size_t i = 0; i < bin_files.size(); ++i) {
        std::vector<PointXYZI> points = load_velodyne_bin(bin_files[i]);
        sensor_msgs::msg::PointCloud2 msg = create_pointcloud_msg(points, timestamps[i]);

        // Create timestamp for the bag
        rclcpp::Time timestamp(static_cast<int64_t>(timestamps[i] * 1e9));
        writer.write(msg, topic_name, timestamp);

        if (i % 100 == 0) {
            std::cout << "Progress: " << i << "/" << bin_files.size() << " frames processed ("
                      << static_cast<int>((i * 100.0) / bin_files.size()) << "%)" << std::endl;
        }
    }

    std::cout << "Saved " << bin_files.size() << " point clouds to " << bag_file << std::endl;
}

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);

    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <velodyne_dir> <timestamp_file> [output_bag_name]" << std::endl;
        std::cerr << "Example: " << argv[0]
                  << " /path/to/sequences/00/velodyne/ /path/to/sequences/00/times.txt kitti_sequence_00" << std::endl;
        rclcpp::shutdown();
        return 1;
    }

    try {
        // Get paths from command line arguments
        const std::string velodyne_dir = argv[1];
        const std::string timestamp_file = argv[2];
        const std::string output_bag = (argc > 3) ? argv[3] : "kitti_sequence";

        std::cout << "Using velodyne directory: " << velodyne_dir << std::endl;
        std::cout << "Using timestamp file: " << timestamp_file << std::endl;
        std::cout << "Output bag name: " << output_bag << std::endl;

        // Load bin files and timestamps
        std::vector<std::string> bin_files = load_bin_files(velodyne_dir);
        std::vector<double> timestamps = load_timestamps(timestamp_file);

        std::cout << "Found " << bin_files.size() << " bin files and " << timestamps.size() << " timestamps."
                  << std::endl;

        // Save to ROS2 bag
        save_to_rosbag(bin_files, timestamps, output_bag);
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        rclcpp::shutdown();
        return 1;
    }

    rclcpp::shutdown();
    return 0;
}