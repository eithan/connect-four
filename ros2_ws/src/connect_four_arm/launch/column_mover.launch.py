"""
column_mover.launch.py

Launches the ColumnMover node.
Run AFTER ur_sim_control.launch.py and ur_moveit.launch.py are up.

Launch:
    ros2 launch connect_four_arm column_mover.launch.py

Test:
    ros2 topic pub --once /connect_four/drop_column std_msgs/Int32 "{data: 3}"
"""

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    column_mover = Node(
        package="connect_four_arm",
        executable="column_mover.py",
        name="column_mover",
        parameters=[{"use_sim_time": True}],
        output="screen",
    )
    return LaunchDescription([column_mover])
