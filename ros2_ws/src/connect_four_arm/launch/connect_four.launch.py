"""
connect_four.launch.py

Single-command startup for the Connect Four simulation stack.

This launcher keeps ownership of the child launch descriptions instead of
spawning nested ``ros2 launch`` subprocesses where possible. That makes
shutdown cleaner and avoids leaving duplicate Gazebo / controller / RViz
processes behind between runs, which was contaminating the ROS graph and
causing TF / controller errors.

Launch:
    ros2 launch connect_four_arm connect_four.launch.py

Test move (separate terminal):
    ros2 topic pub --once /connect_four/drop_column std_msgs/Int32 "{data: 3}"

Full game (separate terminal):
    python3 game_loop.py --ros

Sequencing:
    t=0     Kill stale processes from previous runs
    t=1.5s  Start Xvfb + Gazebo (via connect_four_sim.launch.py)
    t=10s   Start MoveIt2 + RViz2 (via connect_four_moveit.launch.py subprocess)
             - wait_for_robot_description gates move_group and rviz2 startup
    t=12s   Start column_mover
             - polls /compute_ik internally before moving
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    pkg = get_package_share_directory("connect_four_arm")

    # --- Kill stale processes from any previous run ---
    cleanup = [
        ExecuteProcess(cmd=["pkill", "-x", "Xvfb"], output="screen"),
        ExecuteProcess(cmd=["pkill", "-x", "rviz2"], output="screen"),
        ExecuteProcess(cmd=["pkill", "-x", "move_group"], output="screen"),
        ExecuteProcess(cmd=["pkill", "-f", "gz sim"], output="screen"),
        ExecuteProcess(cmd=["pkill", "-f", "ur_sim_control.launch.py"], output="screen"),
        ExecuteProcess(cmd=["pkill", "-f", "ur_moveit.launch.py"], output="screen"),
        ExecuteProcess(cmd=["pkill", "-f", "connect_four_moveit.launch.py"], output="screen"),
        ExecuteProcess(cmd=["pkill", "-f", "column_mover.py"], output="screen"),
        ExecuteProcess(cmd=["pkill", "-f", "controller_manager"], output="screen"),
        ExecuteProcess(cmd=["pkill", "-f", "spawner"], output="screen"),
        ExecuteProcess(cmd=["pkill", "-f", "robot_state_publisher"], output="screen"),
        ExecuteProcess(cmd=["rm", "-f", "/tmp/.X99-lock"], output="screen"),
    ]

    # --- t=1.5s: Xvfb + Gazebo + ros2_control ---
    # Inlined so all Gazebo child processes share this launch context (clean shutdown).
    sim_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg, "launch", "connect_four_sim.launch.py")
        )
    )

    # --- t=10s: MoveIt2 + RViz2 ---
    # Run as a subprocess so its MoveItConfigsBuilder args stay in their own
    # launch context and don't leak into ours.
    moveit_launch = ExecuteProcess(
        cmd=["ros2", "launch", "connect_four_arm", "connect_four_moveit.launch.py"],
        output="screen",
    )

    # --- t=12s: column_mover ---
    # Polls /compute_ik internally before attempting arm motion.
    column_mover_node = Node(
        package="connect_four_arm",
        executable="column_mover.py",
        name="column_mover",
        parameters=[{"use_sim_time": True}],
        output="screen",
    )

    return LaunchDescription(
        cleanup + [
            TimerAction(period=1.5, actions=[sim_launch]),
            TimerAction(period=10.0, actions=[moveit_launch]),
            TimerAction(period=12.0, actions=[column_mover_node]),
        ]
    )
