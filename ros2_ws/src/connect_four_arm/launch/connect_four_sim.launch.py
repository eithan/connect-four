"""
connect_four_sim.launch.py

Replaces the call to ur_simulation_gz/ur_sim_control.launch.py to work
around the ROS 2 Jazzy requirement that robot_description be wrapped in
ParameterValue(value_type=str) when passed as a node parameter.

Starts the headless Gazebo simulation with the UR5e + Robotiq gripper.
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, FindExecutable, IfElseSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg = get_package_share_directory("connect_four_arm")

    world_file       = os.path.join(pkg, "worlds", "connect_four.world")
    description_file = os.path.join(pkg, "urdf",   "ur5e_with_gripper.urdf.xacro")
    controllers_file = os.path.join(pkg, "config",  "connect_four_controllers.yaml")

    # Build URDF string from xacro — must be wrapped in ParameterValue(value_type=str)
    # so ROS 2 does not try to parse the XML as YAML.
    robot_description_content = Command([
        FindExecutable(name="xacro"), " ",
        description_file,
        " name:=ur",
        " ur_type:=ur5e",
        " tf_prefix:=",
        " safety_limits:=true",
        " safety_pos_margin:=0.15",
        " safety_k_position:=20",
        " simulation_controllers:=", controllers_file,
    ])
    robot_description = {
        "robot_description": ParameterValue(robot_description_content, value_type=str)
    }

    # Virtual framebuffer — Gazebo needs a display even in headless mode
    xvfb = ExecuteProcess(
        cmd=["Xvfb", ":99", "-screen", "0", "1024x768x24"],
        output="log",
    )

    # Publish URDF to /robot_description
    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="both",
        parameters=[{"use_sim_time": True}, robot_description],
    )

    # Gazebo Harmonic (headless, with custom world)
    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [FindPackageShare("ros_gz_sim"), "/launch/gz_sim.launch.py"]
        ),
        launch_arguments={
            "gz_args": f" -s -r -v 4 {world_file}",
        }.items(),
    )

    # Bridge /clock so ROS nodes receive sim time
    gz_clock_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=["/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock"],
        output="screen",
    )

    # Spawn the robot model into Gazebo
    gz_spawn_robot = Node(
        package="ros_gz_sim",
        executable="create",
        output="screen",
        arguments=[
            "-string", robot_description_content,
            "-name", "ur",
            "-allow_renaming", "true",
        ],
    )

    # Spawn joint_state_broadcaster
    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster", "-c", "/controller_manager"],
    )

    # Spawn joint_trajectory_controller
    jtc_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_trajectory_controller", "-c", "/controller_manager"],
    )

    # Bridge overhead camera topics from Gazebo to ROS
    gz_camera_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=[
            "/overhead_camera/image@sensor_msgs/msg/Image[gz.msgs.Image",
            "/overhead_camera/depth_image@sensor_msgs/msg/Image[gz.msgs.Image",
            "/overhead_camera/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo",
            "/overhead_camera/points@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked",
        ],
        output="screen",
    )

    # Static TF for overhead camera frame (world -> overhead_camera_link)
    static_tf_camera = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        arguments=["0.55", "0.0", "1.2", "0", "1.5708", "0", "world", "overhead_camera_link"],
    )

    return LaunchDescription([
        xvfb,
        TimerAction(period=1.0, actions=[
            robot_state_publisher,
            gz_sim,
            gz_clock_bridge,
            static_tf_camera,
        ]),
        TimerAction(period=5.0, actions=[gz_spawn_robot]),
        TimerAction(period=8.0, actions=[joint_state_broadcaster_spawner, jtc_spawner]),
        TimerAction(period=10.0, actions=[gz_camera_bridge]),
    ])
