"""
connect_four_sim.launch.py

Starts the headless Gazebo simulation (no GUI window needed).
Xvfb is started automatically as a virtual framebuffer for Gazebo rendering.

Launch:
    ros2 launch connect_four_arm connect_four_sim.launch.py
"""

from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction


def generate_launch_description():
    # Virtual framebuffer — Gazebo needs a display even in headless mode
    xvfb = ExecuteProcess(
        cmd=["Xvfb", ":99", "-screen", "0", "1024x768x24"],
        output="log",
    )

    # Gazebo + ros2_control + joint controllers (headless, no GUI)
    gazebo_sim = ExecuteProcess(
        cmd=[
            "ros2", "launch", "ur_simulation_gz", "ur_sim_control.launch.py",
            "ur_type:=ur5e", "gazebo_gui:=false", "use_sim_time:=true",
        ],
        additional_env={"DISPLAY": ":99", "GZ_IP": "127.0.0.1"},
        output="screen",
    )

    return LaunchDescription([
        xvfb,
        TimerAction(period=1.0, actions=[gazebo_sim]),  # let Xvfb start first
    ])
