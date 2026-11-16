import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    share_dir = get_package_share_directory('mapper3d')
    rviz_config = os.path.join(share_dir, 'rviz', 'mapper3d_config.rviz')
    params_file = os.path.join(share_dir, 'config', 'mapper3d.yaml')

    return LaunchDescription([
        DeclareLaunchArgument(
            'params_file',
            default_value=params_file,
            description='YAML file with the parameters of mapper3d_node',
        ),
        Node(
            package='mapper3d',
            executable='mapper3d_node',
            name='mapper3d_node',
            output='screen',
            parameters=[LaunchConfiguration('params_file')],
            # prefix=['xterm -geometry 120x40 -bg black -fg white -fa UbuntuMono-RI -fs 16 -e lldb --'],
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', rviz_config],
        ),
    ])
