from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='mapper3d',
            executable='mapper3d_node',
            name='mapper3d_node',
            output='screen',
            # prefix=['xterm -geometry 120x40 -bg black -fg white -fa UbuntuMono-RI -fs 20 -e lldb --'],
        )
    ])