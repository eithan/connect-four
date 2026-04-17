#!/usr/bin/env python3
"""
column_mover.py — Move UR5e above a Connect Four column.

Subscribes to /connect_four/drop_column (std_msgs/Int32, values 0-6).
Acts as a MoveIt2 CLIENT to the already-running move_group node.

On startup, publishes RViz2 markers showing the board and column targets.
The board is visual-only (not a collision object) until arm poses are confirmed.

Usage:
    ros2 run connect_four_arm column_mover.py
    ros2 topic pub --once /connect_four/drop_column std_msgs/Int32 "{data: 3}"
"""

import math
import time
import threading
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from std_msgs.msg import Int32, ColorRGBA
from geometry_msgs.msg import Point, Vector3
from visualization_msgs.msg import Marker, MarkerArray
from pymoveit2 import MoveIt2

try:
    from pymoveit2 import MoveIt2State
    _HAS_STATE = True
except ImportError:
    _HAS_STATE = False


# UR5e joint names (order must match URDF)
UR5E_JOINTS = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

# --- Board geometry (Hasbro-style, in meters) ---
BOARD_WIDTH   = 0.292   # 7 columns × 42 mm
BOARD_HEIGHT  = 0.254   # 6 rows  × 42 mm
BOARD_DEPTH   = 0.064
BOARD_X       = 0.65    # distance from arm base to board center
BOARD_Z_BASE  = 0.0     # board sits on table at z=0

BOARD_CENTER = [BOARD_X, 0.0, BOARD_Z_BASE + BOARD_HEIGHT / 2]

# Robotiq 2F-85 TCP offset: distance from tool0 flange to gripper fingertip plane
# (UR-to-Robotiq adapter 11 mm + robotiq_base to fingertip 98 mm = 109 mm,
# measured along tool0 z-axis when arm points straight down).
# tool0 must be positioned this much higher than the desired fingertip target z.
GRIPPER_TCP_OFFSET = 0.109

# Drop height: gripper fingertips hover DROP_CLEARANCE above the board top.
# tool0 z = (BOARD_TOP + DROP_CLEARANCE) + GRIPPER_TCP_OFFSET
DROP_CLEARANCE = 0.05           # 50 mm above board top (fingertip target)
DROP_Z = BOARD_Z_BASE + BOARD_HEIGHT + DROP_CLEARANCE + GRIPPER_TCP_OFFSET  # 0.413 m

# Column centers: 7 evenly spaced, symmetric around y=0
COL_SPACING = BOARD_WIDTH / 7  # ~41.7 mm
COLUMN_POSES = [
    (BOARD_X, (3 - i) * COL_SPACING, DROP_Z)
    for i in range(7)
]

# Quaternion: tool0 pointing straight down (180° rotation around Y)
QUAT_DOWN = [0.0, 1.0, 0.0, 0.0]

MOVE_TIMEOUT = 20.0  # seconds to wait for a move to complete

# Safe parked configuration — arm retracted upright, clear of the board.
# joint order: shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3
HOME_JOINTS = [0.0, -1.5708, 0.0, -1.5708, 0.0, 0.0]


class ColumnMover(Node):
    def __init__(self):
        super().__init__("column_mover")

        # pymoveit2 action/service callbacks live in this group.
        cb_group = ReentrantCallbackGroup()
        # The drop_column subscription gets its OWN group so that pymoveit2's
        # action callbacks can never starve it — they compete for the same
        # executor threads only within their own group.
        self._sub_cb_group = ReentrantCallbackGroup()

        # use_move_group_action=False: plan via /plan_kinematic_path service,
        # execute via /execute_trajectory action (MoveGroup routes to
        # joint_trajectory_controller per connect_four_moveit_controllers.yaml).
        # follow_joint_trajectory_action_name is intentionally omitted —
        # newer pymoveit2 ignores it and uses /execute_trajectory regardless.
        self._moveit2 = MoveIt2(
            node=self,
            joint_names=UR5E_JOINTS,
            base_link_name="base_link",
            end_effector_name="tool0",
            group_name="ur_manipulator",
            use_move_group_action=True,
            callback_group=cb_group,
        )
        self._moveit2.max_velocity_scaling_factor = 0.5
        self._moveit2.max_acceleration_scaling_factor = 0.25
        self._moveit2.planner_id = "RRTConnect"
        self._moveit2.allowed_planning_time = 10.0

        self._lock = threading.Lock()

        # Joint configs precomputed at startup via /compute_ik.
        # Populated by _precompute_column_ik(); moves use these instead of
        # move_to_pose so OMPL never has to search in Cartesian space.
        self._column_joints: dict[int, list[float]] = {}

        # Transient Local (latched): RViz receives the last message even if it
        # subscribes after markers are first published.
        _marker_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self._marker_pub = self.create_publisher(
            MarkerArray, "/connect_four/markers", _marker_qos
        )
        self._markers: MarkerArray | None = None  # cached after first build

        self.create_subscription(
            Int32,
            "/connect_four/drop_column",
            self._on_column,
            10,
            callback_group=self._sub_cb_group,
        )

        self._busy = False   # True while a move is in progress

        # Publish the visual board immediately so RViz can show it even while
        # MoveIt is still starting. Poll separately for MoveIt readiness before
        # attempting the first arm motion.
        self._publish_markers()
        self._setup_timer = self.create_timer(
            1.0, self._wait_for_moveit, callback_group=cb_group
        )
        self._marker_timer = self.create_timer(
            2.0, self._republish_markers, callback_group=cb_group
        )

        self.get_logger().info(
            "ColumnMover ready — waiting for MoveIt2 (/compute_ik)..."
        )

    def _wait_for_moveit(self):
        """Poll until /compute_ik service exists, then initialise."""
        services = dict(self.get_service_names_and_types())
        if "/compute_ik" not in services:
            return  # try again next tick
        self._setup_timer.cancel()
        self.get_logger().info("MoveIt2 ready — precomputing column IK")
        threading.Thread(target=self._init_sequence, daemon=True).start()

    def _init_sequence(self):
        """Precompute IK for all columns, then move home."""
        self._precompute_column_ik()
        self._go_home()

    @staticmethod
    def _normalize_joints(joints, reference):
        """Wrap each joint angle to the equivalent value closest to reference."""
        result = []
        for j, r in zip(joints, reference):
            n = round((r - j) / (2 * math.pi))
            result.append(j + 2 * math.pi * n)
        return result

    def _call_ik(self, ik_cli, x, y, z, seed):
        """Call /compute_ik once with a given seed; return joint list or None."""
        from moveit_msgs.srv import GetPositionIK
        from geometry_msgs.msg import PoseStamped

        req = GetPositionIK.Request()
        req.ik_request.group_name = "ur_manipulator"
        req.ik_request.avoid_collisions = True

        ps = PoseStamped()
        ps.header.frame_id = "base_link"
        ps.pose.position.x = x
        ps.pose.position.y = y
        ps.pose.position.z = z
        ps.pose.orientation.x = float(QUAT_DOWN[0])
        ps.pose.orientation.y = float(QUAT_DOWN[1])
        ps.pose.orientation.z = float(QUAT_DOWN[2])
        ps.pose.orientation.w = float(QUAT_DOWN[3])
        req.ik_request.pose_stamped = ps
        req.ik_request.robot_state.joint_state.name = list(UR5E_JOINTS)
        req.ik_request.robot_state.joint_state.position = list(seed)

        future = ik_cli.call_async(req)
        deadline = time.time() + 5.0
        while not future.done() and time.time() < deadline:
            time.sleep(0.05)

        if not future.done() or future.result().error_code.val != 1:
            return None

        js = future.result().solution.joint_state
        pos_map = dict(zip(js.name, js.position))
        joints = [pos_map.get(j) for j in UR5E_JOINTS]
        return None if None in joints else joints

    def _precompute_column_ik(self):
        """Resolve joint configs for every column using /compute_ik.

        Calling move_to_configuration instead of move_to_pose removes the
        Cartesian-orientation constraint from OMPL's search, making planning
        deterministic: the goal is a single point in C-space rather than the
        full IK-solution set that OMPL must sample.

        KDL (the default IK solver) ignores seeds and may return any valid
        solution, including "reach-from-behind" configs far from home.  We try
        several seeds and keep whichever normalised result is closest to
        HOME_JOINTS in joint space, while also requiring that q1 (shoulder_pan)
        points toward the column (not backward).
        """
        from moveit_msgs.srv import GetPositionIK

        ik_cli = self.create_client(GetPositionIK, "/compute_ik")
        if not ik_cli.wait_for_service(timeout_sec=10.0):
            self.get_logger().warn(
                "IK service unavailable — column moves will use pose planning"
            )
            return

        for col, (x, y, z) in enumerate(COLUMN_POSES):
            col_pan = math.atan2(y, x)

            # Seeds that bracket the expected elbow-up, tool-down configuration.
            # q5 = ±π/2 is what the UR5e needs for tool-pointing-down; two sign
            # variants are included because different columns need different chirality.
            seeds = [
                [col_pan, -2.1,    2.0,  -1.5, -math.pi / 2, 0.0],
                [col_pan, -2.1,    2.0,  -1.5,  math.pi / 2, 0.0],
                [col_pan, -2.2,    1.8,  -1.2, -math.pi / 2, 0.0],
                [col_pan, -2.0,    1.5,  -1.0,  math.pi / 2, 0.0],
                [col_pan, -1.5708, 0.0, -1.5708, 0.0,         0.0],  # home-like
            ]

            best_joints = None
            best_dist = float("inf")

            for seed in seeds:
                raw = self._call_ik(ik_cli, x, y, z, seed)
                if raw is None:
                    continue

                # Wrap every joint to the equivalent angle nearest HOME_JOINTS.
                normalised = self._normalize_joints(raw, HOME_JOINTS)

                # Reject "reach-from-behind": shoulder_pan must point toward column.
                if abs(normalised[0] - col_pan) > math.pi / 2:
                    continue

                dist = sum((a - b) ** 2 for a, b in zip(normalised, HOME_JOINTS))
                if dist < best_dist:
                    best_dist = dist
                    best_joints = normalised

            if best_joints is not None:
                self._column_joints[col] = best_joints
                self.get_logger().info(
                    f"IK col {col}: [{', '.join(f'{j:.3f}' for j in best_joints)}]"
                )
            else:
                self.get_logger().warn(
                    f"No near-home IK solution for col {col} — will use pose planning"
                )

        self.get_logger().info(
            f"Column IK precomputed: {len(self._column_joints)}/7 columns"
        )

    def _publish_markers(self):
        markers = MarkerArray()

        def make_header():
            from std_msgs.msg import Header
            h = Header()
            h.frame_id = "base_link"
            h.stamp.sec = 0      # timestamp=0 means "always valid" in RViz2
            h.stamp.nanosec = 0
            return h

        # Semi-transparent blue box — the board (visual only, not a collision object)
        board = Marker()
        board.header = make_header()
        board.ns = "connect_four"
        board.id = 0
        board.type = Marker.CUBE
        board.action = Marker.ADD
        board.pose.position = Point(
            x=BOARD_CENTER[0], y=BOARD_CENTER[1], z=BOARD_CENTER[2]
        )
        board.pose.orientation.w = 1.0
        board.scale = Vector3(x=BOARD_DEPTH, y=BOARD_WIDTH, z=BOARD_HEIGHT)
        board.color = ColorRGBA(r=0.1, g=0.3, b=0.9, a=0.5)
        markers.markers.append(board)

        # Yellow spheres: one per column drop point
        for col, (x, y, z) in enumerate(COLUMN_POSES):
            m = Marker()
            m.header = make_header()
            m.ns = "connect_four_columns"
            m.id = col + 1
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position = Point(x=x, y=y, z=z)
            m.pose.orientation.w = 1.0
            m.scale = Vector3(x=0.025, y=0.025, z=0.025)
            m.color = ColorRGBA(r=1.0, g=0.85, b=0.0, a=0.9)
            markers.markers.append(m)

        self._markers = markers
        self._marker_pub.publish(self._markers)
        self.get_logger().info("Connect Four board markers published")

    def _republish_markers(self):
        if self._markers is not None:
            self._marker_pub.publish(self._markers)

    def _on_column(self, msg: Int32):
        col = msg.data
        self.get_logger().info(f"Received drop_column request: {col}")
        if not 0 <= col <= 6:
            self.get_logger().error(f"Column {col} out of range — must be 0-6")
            return
        if self._busy:
            self.get_logger().warn(f"Arm busy — ignoring column {col} request")
            return
        threading.Thread(target=self._move, args=(col,), daemon=True).start()

    def _wait_executed(self):
        """Wait for the current move to complete using state polling only.

        Does NOT call wait_until_executed() — in some pymoveit2 versions that
        method calls rclpy.spin_once() internally, which corrupts the
        MultiThreadedExecutor and silently kills all future subscription and
        timer callbacks (including _on_column).

        Phase 1: wait up to 2 s for state to leave IDLE, confirming the goal
                 was accepted by the action server.
        Phase 2: wait up to MOVE_TIMEOUT for state to return to IDLE,
                 confirming execution completed.
        """
        if not _HAS_STATE:
            time.sleep(MOVE_TIMEOUT)
            return

        # Phase 1 — wait for goal to be accepted (state leaves IDLE)
        deadline = time.time() + 2.0
        while self._moveit2.query_state() == MoveIt2State.IDLE and time.time() < deadline:
            time.sleep(0.05)

        # Phase 2 — wait for execution to finish (state returns to IDLE)
        start = time.time()
        while (
            self._moveit2.query_state() != MoveIt2State.IDLE
            and time.time() - start < MOVE_TIMEOUT
        ):
            time.sleep(0.1)

    def _go_home(self):
        """Move arm to the safe home configuration."""
        with self._lock:
            self._busy = True
            try:
                self.get_logger().info("Moving to home position")
                self._moveit2.move_to_configuration(HOME_JOINTS)
                self._wait_executed()
                self.get_logger().info("Home position reached")
            except Exception as exc:
                self.get_logger().error(f"Home move failed: {exc}")
            finally:
                self._busy = False

    def _move(self, col: int):
        with self._lock:
            self._busy = True
            try:
                x, y, z = COLUMN_POSES[col]
                self.get_logger().info(
                    f"Moving to column {col}  ({x:.3f}, {y:.3f}, {z:.3f})"
                )

                if col in self._column_joints:
                    # Joint-space goal: OMPL searches a single point in C-space,
                    # not a Cartesian IK-solution set. Always succeeds for reachable
                    # configurations — no probabilistic failures.
                    self._moveit2.move_to_configuration(self._column_joints[col])
                else:
                    # Fallback for columns where IK precomputation failed.
                    self.get_logger().warn(
                        f"No precomputed IK for col {col} — using pose planning"
                    )
                    self._moveit2.move_to_pose(
                        position=[x, y, z],
                        quat_xyzw=QUAT_DOWN,
                        cartesian=False,
                    )

                self._wait_executed()
                self.get_logger().info(f"Column {col} reached — returning home")

                self._moveit2.move_to_configuration(HOME_JOINTS)
                self._wait_executed()
                self.get_logger().info("Home position reached")
            except Exception as exc:
                self.get_logger().error(f"Column {col} move failed: {exc}")
            finally:
                self._busy = False


def main(args=None):
    rclpy.init(args=args)
    node = ColumnMover()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        executor.shutdown()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
