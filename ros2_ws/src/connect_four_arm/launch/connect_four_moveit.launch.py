"""
connect_four_moveit.launch.py

Starts a local MoveIt2 stack for the UR simulation using the Connect Four RViz
profile and a controller mapping that matches the Gazebo simulation.
"""

import os
from pathlib import Path

from launch import LaunchDescription
from launch.actions import RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg = get_package_share_directory("connect_four_arm")
    rviz_config = os.path.join(pkg, "config", "connect_four.rviz")
    warehouse_sqlite_path = os.path.expanduser("~/.ros/warehouse_ros.sqlite")

    moveit_config = (
        MoveItConfigsBuilder(robot_name="ur", package_name="ur_moveit_config")
        .robot_description_semantic(
            Path(pkg) / "config" / "ur5e_with_gripper.srdf.xacro", {"name": "ur"}
        )
        .trajectory_execution(
            Path(pkg) / "config" / "connect_four_moveit_controllers.yaml",
            moveit_manage_controllers=True,
        )
        .to_moveit_configs()
    )

    warehouse_ros_config = {
        "warehouse_plugin": "warehouse_ros_sqlite::DatabaseConnection",
        "warehouse_host": warehouse_sqlite_path,
    }

    wait_robot_description = Node(
        package="ur_robot_driver",
        executable="wait_for_robot_description",
        output="screen",
    )

    move_group = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[
            moveit_config.to_dict(),
            warehouse_ros_config,
            {
                "use_sim_time": True,
                "publish_robot_description_semantic": True,
            },
        ],
    )

    rviz2 = Node(
        package="rviz2",
        executable="rviz2",
        name="connect_four_rviz",
        arguments=["-d", rviz_config],
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
            moveit_config.planning_pipelines,
            moveit_config.joint_limits,
            moveit_config.trajectory_execution,
            warehouse_ros_config,
            {"use_sim_time": True},
        ],
        output="screen",
    )

    return LaunchDescription(
        [
            wait_robot_description,
            RegisterEventHandler(
                OnProcessExit(
                    target_action=wait_robot_description,
                    on_exit=[move_group, rviz2],
                )
            ),
        ]
    )
