"""
connect_four_sim.launch.py

Starts the headless Gazebo simulation (no GUI window needed).
Xvfb is started automatically as a virtual framebuffer for Gazebo rendering.

Launch:
    ros2 launch connect_four_arm connect_four_sim.launch.py
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction


def generate_launch_description():
    pkg = get_package_share_directory("connect_four_arm")

    world_file       = os.path.join(pkg, "worlds", "connect_four.world")
    description_file = os.path.join(pkg, "urdf",   "ur5e_with_gripper.urdf.xacro")
    controllers_file = os.path.join(pkg, "config",  "connect_four_controllers.yaml")

    # Virtual framebuffer — Gazebo needs a display even in headless mode
    xvfb = ExecuteProcess(
        cmd=["Xvfb", ":99", "-screen", "0", "1024x768x24"],
        output="log",
    )

    # Gazebo + ros2_control + joint controllers (headless, no GUI).
    # world_file:       custom scene (table, board, pieces)
    # description_file: UR5e + Robotiq 2F-85 combined xacro
    # controllers_file: ur_controllers + gripper_controller
    gazebo_sim = ExecuteProcess(
        cmd=[
            "ros2", "launch", "ur_simulation_gz", "ur_sim_control.launch.py",
            "ur_type:=ur5e", "gazebo_gui:=false", "launch_rviz:=false",
            "initial_joint_controller:=joint_trajectory_controller",
            "use_sim_time:=true",
            f"world_file:={world_file}",
            f"description_file:={description_file}",
            f"controllers_file:={controllers_file}",
        ],
        additional_env={"DISPLAY": ":99", "GZ_IP": "127.0.0.1"},
        output="screen",
    )

    return LaunchDescription([
        xvfb,
        TimerAction(period=1.0, actions=[gazebo_sim]),  # let Xvfb start first
    ])
