"""
connect_four_moveit.launch.py

Starts MoveIt2 (with RViz2) and the column mover node.
Run AFTER connect_four_sim.launch.py is up and stable.

The column mover waits 12 seconds for MoveIt2 to finish initializing
before it tries to connect — adjust the timer if your machine is slower.

Launch:
    ros2 launch connect_four_arm connect_four_moveit.launch.py

Test:
    ros2 topic pub --once /connect_four/drop_column std_msgs/Int32 "{data: 3}"
"""

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    moveit = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            get_package_share_directory("ur_moveit_config"),
            "/launch/ur_moveit.launch.py",
        ]),
        launch_arguments={
            "ur_type": "ur5e",
            "launch_rviz": "false",  # we launch RViz2 ourselves for clean shutdown
            "use_sim_time": "true",
        }.items(),
    )

    rviz2 = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        arguments=["-d", "/opt/ros/jazzy/share/ur_moveit_config/config/moveit.rviz"],
        parameters=[{"use_sim_time": True}],
        output="screen",
        on_exit=Shutdown(),  # closing RViz2 window shuts down this terminal cleanly
    )

    return LaunchDescription([moveit, rviz2])
