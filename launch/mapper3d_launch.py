import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    rviz_config = os.path.join(
        get_package_share_directory('mapper3d'), 'rviz', 'mapper3d_config.rviz')

    return LaunchDescription([
        Node(
            package='mapper3d',
            executable='mapper3d_node',
            name='mapper3d_node',
            output='screen',
            # prefix=['xterm -geometry 120x40 -bg black -fg white -fa UbuntuMono-RI -fs 16 -e lldb --'],
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', rviz_config],
        ),
    ])
