"""
view_panda.launch.py

Loads the Panda arm URDF and visualizes it in RViz2 with a joint slider GUI.
This is a learning exercise — no Gazebo or MoveIt2 needed.

Launch:
    ros2 launch connect_four_arm view_panda.launch.py
"""

from pathlib import Path
from launch import LaunchDescription
from launch.actions import Shutdown
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


PANDA_URDF = "/opt/ros/jazzy/share/moveit_resources_panda_description/urdf/panda.urdf"


def generate_launch_description():
    urdf = Path(PANDA_URDF).read_text()
    rviz_config = Path(get_package_share_directory("connect_four_arm")) / "config" / "panda.rviz"

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        parameters=[{"robot_description": urdf}],
    )

    joint_state_publisher_gui = Node(
        package="joint_state_publisher_gui",
        executable="joint_state_publisher_gui",
        name="joint_state_publisher_gui",
    )

    rviz2 = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        arguments=["-d", str(rviz_config)],
        on_exit=Shutdown(),
    )

    return LaunchDescription([
        robot_state_publisher,
        joint_state_publisher_gui,
        rviz2,
    ])
