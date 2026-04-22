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

import json
import math
import time
import threading
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from rclpy.action import ActionClient
from std_msgs.msg import Int32, String, ColorRGBA
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Point, Vector3
from visualization_msgs.msg import Marker, MarkerArray
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration as RosDuration
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

MOVE_TIMEOUT = 90.0  # seconds to wait for a move to complete
MAX_PICK_RETRIES = 2  # re-attempts if gripper closes in air

# --- Supply tray geometry ---
TRAY_RED_X    = 0.50
TRAY_YELLOW_X = 0.42
TRAY_Y_START  = -0.126
TRAY_Y_STEP   = 0.042
TRAY_PIECES = {
    'red':    [(TRAY_RED_X,    TRAY_Y_START + i * TRAY_Y_STEP) for i in range(7)],
    'yellow': [(TRAY_YELLOW_X, TRAY_Y_START + i * TRAY_Y_STEP) for i in range(7)],
}

# Pick heights (tool0 z)
# PICK_APPROACH_Z matches DROP_Z so the tray-to-column transit (Step 6) stays
# at a constant height above the board top (0.254 m), preventing the arm from
# dipping into the board's open column slots during the joint-space trajectory.
PICK_APPROACH_Z = DROP_Z   # 0.413 m — same as column drop height
PICK_MID_Z      = 0.165    # intermediate descent: fingertips 56mm above tray (well clear)
PICK_GRASP_Z    = 0.134    # fingertips 25mm above tray: gives safe clearance vs overshoot

# Gripper joint positions (robotiq_85_left_knuckle_joint, radians)
GRIPPER_OPEN          = 0.0
GRIPPER_PIECE         = 0.55   # ~33mm gap; tune empirically
GRIPPER_MOVE_DURATION = 1.0    # seconds

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
        self._moveit2.max_velocity = 1.0
        self._moveit2.max_acceleration = 0.5
        self._moveit2.pipeline_id = "pilz_industrial_motion_planner"
        self._moveit2.planner_id = "PTP"
        self._moveit2.allowed_planning_time = 10.0

        self._lock = threading.Lock()

        # Joint configs precomputed at startup via /compute_ik.
        # Populated by _precompute_column_ik(); moves use these instead of
        # move_to_pose so OMPL never has to search in Cartesian space.
        self._column_joints: dict[int, list[float]] = {}

        # Tray IK: keyed by "color_idx" (e.g. "red_0", "yellow_3")
        self._tray_joints: dict[str, list[float]] = {}
        # Tray mid IK: same x,y at PICK_MID_Z (intermediate descent waypoint)
        self._tray_mid_joints: dict[str, list[float]] = {}
        # Tray grasp IK: same x,y as tray but at PICK_GRASP_Z height
        self._tray_grasp_joints: dict[str, list[float]] = {}
        # Next piece index to pick for each color (consumed in order 0..6)
        self._next_piece: dict[str, int] = {'red': 0, 'yellow': 0}
        # Latest piece positions from the overhead camera detector
        self._detected_pieces: dict[str, list] = {'red': [], 'yellow': []}

        # Persistent IK service client — reused during pick-place for live re-targeting
        from moveit_msgs.srv import GetPositionIK
        self._ik_cli = self.create_client(
            GetPositionIK, "/compute_ik", callback_group=cb_group
        )

        # Direct arm action client — bypasses MoveIt/Pilz planning entirely.
        # Trajectories are sent straight to the JTC with a fixed sim-time duration,
        # avoiding the Pilz velocity-scaling bug (23s trajectories instead of ~2s).
        self._arm_action = ActionClient(
            self,
            FollowJointTrajectory,
            '/joint_trajectory_controller/follow_joint_trajectory',
            callback_group=cb_group,
        )

        # Direct gripper action client (bypasses MoveIt; pymoveit2's gripper
        # support was removed from the installed version)
        self._gripper_action = ActionClient(
            self,
            FollowJointTrajectory,
            '/gripper_controller/follow_joint_trajectory',
            callback_group=cb_group,
        )

        # Latest joint positions from /joint_states — polled in _move_arm_joints
        # to detect when the arm has physically reached the target, bypassing
        # the unreliable JTC result_future for large moves.
        self._joint_pos: dict[str, float] = {}
        self.create_subscription(
            JointState,
            '/joint_states',
            self._on_joint_state,
            10,
            callback_group=cb_group,
        )

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

        self.create_subscription(
            String,
            '/connect_four/pick_and_place',
            self._on_pick_and_place,
            10,
            callback_group=self._sub_cb_group,
        )

        self.create_subscription(
            String,
            '/connect_four/piece_positions',
            self._on_piece_positions,
            10,
            callback_group=cb_group,
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
        """Precompute IK for all columns and tray positions, then move home."""
        self._precompute_column_ik()
        self._precompute_tray_ik()
        self._go_home()

    @staticmethod
    def _normalize_joints(joints, reference):
        """Wrap each joint angle to the equivalent value closest to reference."""
        result = []
        for j, r in zip(joints, reference):
            n = round((r - j) / (2 * math.pi))
            result.append(j + 2 * math.pi * n)
        return result

    def _call_ik(self, ik_cli, x, y, z, seed, avoid_collisions=True):
        """Call /compute_ik once with a given seed; return joint list or None."""
        from moveit_msgs.srv import GetPositionIK
        from geometry_msgs.msg import PoseStamped

        req = GetPositionIK.Request()
        req.ik_request.group_name = "ur_manipulator"
        req.ik_request.avoid_collisions = avoid_collisions

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

    def _precompute_tray_ik(self):
        """Resolve joint configs for every tray position using /compute_ik.

        Uses the same seed strategy as _precompute_column_ik, adapted for
        the tray x-positions (0.42 and 0.50), which are both within UR5e reach.
        """
        from moveit_msgs.srv import GetPositionIK

        ik_cli = self.create_client(GetPositionIK, "/compute_ik")
        if not ik_cli.wait_for_service(timeout_sec=10.0):
            self.get_logger().warn(
                "IK service unavailable — tray IK precomputation skipped"
            )
            return

        total = 0
        for color, positions in TRAY_PIECES.items():
            for idx, (px, py) in enumerate(positions):
                key = f"{color}_{idx}"
                col_pan = math.atan2(py, px)

                seeds = [
                    [col_pan, -2.1,    2.0,  -1.5, -math.pi / 2, 0.0],
                    [col_pan, -2.1,    2.0,  -1.5,  math.pi / 2, 0.0],
                    [col_pan, -2.2,    1.8,  -1.2, -math.pi / 2, 0.0],
                    [col_pan, -2.0,    1.5,  -1.0,  math.pi / 2, 0.0],
                    [col_pan, -1.5708, 0.0, -1.5708, 0.0,         0.0],
                ]

                best_joints = None
                best_dist = float("inf")

                for seed in seeds:
                    raw = self._call_ik(ik_cli, px, py, PICK_APPROACH_Z, seed)
                    if raw is None:
                        continue
                    normalised = self._normalize_joints(raw, HOME_JOINTS)
                    if abs(normalised[0] - col_pan) > math.pi / 2:
                        continue
                    dist = sum((a - b) ** 2 for a, b in zip(normalised, HOME_JOINTS))
                    if dist < best_dist:
                        best_dist = dist
                        best_joints = normalised

                if best_joints is not None:
                    self._tray_joints[key] = best_joints
                    total += 1
                    self.get_logger().info(
                        f"IK tray {key}: [{', '.join(f'{j:.3f}' for j in best_joints)}]"
                    )

                    # Also compute grasp-height IK. First try collision-free;
                    # fall back without collision avoidance if needed (gripper
                    # can be near table at grasp height — ValidateSolution is
                    # disabled in Pilz so the trajectory will still execute).
                    grasp_raw = self._call_ik(ik_cli, px, py, PICK_GRASP_Z, best_joints)
                    if grasp_raw is None:
                        self.get_logger().warn(
                            f"Grasp IK {key}: collision-free failed, "
                            "retrying with avoid_collisions=False"
                        )
                        grasp_raw = self._call_ik(
                            ik_cli, px, py, PICK_GRASP_Z, best_joints,
                            avoid_collisions=False
                        )
                    if grasp_raw is not None:
                        grasp_norm = self._normalize_joints(grasp_raw, best_joints)
                        self._tray_grasp_joints[key] = grasp_norm
                    else:
                        self.get_logger().warn(
                            f"No grasp IK for {key} — will reuse approach joints"
                        )
                        self._tray_grasp_joints[key] = best_joints

                    # Mid approach IK — seeded from approach joints so wrist
                    # configuration matches, keeping the short final descent vertical.
                    mid_raw = self._call_ik(
                        ik_cli, px, py, PICK_MID_Z, best_joints,
                        avoid_collisions=False
                    )
                    if mid_raw is not None:
                        self._tray_mid_joints[key] = self._normalize_joints(mid_raw, best_joints)
                    else:
                        self._tray_mid_joints[key] = best_joints
                else:
                    self.get_logger().warn(
                        f"No IK solution for tray position {key} ({px:.2f}, {py:.3f})"
                    )

        self.get_logger().info(
            f"Tray IK precomputed: {total}/14 positions (approach + mid + grasp)"
        )

    def _move_gripper(self, position: float, duration: float = GRIPPER_MOVE_DURATION):
        """Send a FollowJointTrajectory goal to the gripper controller.

        Polls future.done() (never calls spin_once) to stay safe with
        MultiThreadedExecutor.  Times out after 10 s.
        """
        if not self._gripper_action.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("Gripper action server not available")
            return

        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = ['robotiq_85_left_knuckle_joint']
        pt = JointTrajectoryPoint()
        pt.positions = [float(position)]
        pt.time_from_start = RosDuration(sec=int(duration),
                                         nanosec=int((duration % 1) * 1e9))
        goal.trajectory.points = [pt]

        future = self._gripper_action.send_goal_async(goal)
        deadline = time.time() + 30.0
        while not future.done() and time.time() < deadline:
            time.sleep(0.05)

        if not future.done():
            self.get_logger().error("Gripper goal send timed out")
            return

        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error("Gripper goal rejected")
            return

        result_future = goal_handle.get_result_async()
        deadline = time.time() + 30.0
        while not result_future.done() and time.time() < deadline:
            time.sleep(0.05)

        if not result_future.done():
            self.get_logger().error("Gripper execution timed out")

    def _on_joint_state(self, msg: JointState):
        for name, pos in zip(msg.name, msg.position):
            self._joint_pos[name] = pos

    def _move_arm_joints(self, target_joints: list, sim_duration: float = 4.0,
                         tolerance: float = 0.15, wall_timeout: float = None) -> bool:
        """Send a trajectory directly to the arm JTC and wait until the arm
        physically reaches the target by polling /joint_states.

        Bypasses waiting on the JTC result_future (which is unreliable for large
        moves due to Gazebo PID lag causing goal-tolerance miss at trajectory end).
        Instead, we watch actual joint positions and return as soon as all joints
        are within *tolerance* radians of the target.

        sim_duration is in sim-seconds.  Wall timeout is generous to handle low RTF.
        """
        if not self._arm_action.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("Arm JTC action server not available")
            return False

        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = list(UR5E_JOINTS)
        goal.trajectory.header.stamp = self.get_clock().now().to_msg()
        pt = JointTrajectoryPoint()
        pt.positions = [float(j) for j in target_joints]
        pt.velocities = [0.0] * len(UR5E_JOINTS)
        pt.time_from_start = RosDuration(
            sec=int(sim_duration),
            nanosec=int((sim_duration % 1) * 1e9),
        )
        goal.trajectory.points = [pt]

        send_done = threading.Event()
        send_future = self._arm_action.send_goal_async(goal)
        send_future.add_done_callback(lambda _: send_done.set())
        if not send_done.wait(timeout=30.0):
            self.get_logger().error("Arm JTC goal send timed out")
            return False

        goal_handle = send_future.result()
        if not goal_handle.accepted:
            self.get_logger().error("Arm JTC goal rejected")
            return False

        if wall_timeout is None:
            wall_timeout = sim_duration * 15.0 + 60.0
        start_wall = time.time()
        max_err = float('inf')
        min_err_seen = float('inf')
        last_improvement = time.time()
        STALL_TIMEOUT = 10.0  # abort if error doesn't improve by 0.005 rad in this many seconds

        while time.time() - start_wall < wall_timeout:
            if self._joint_pos:
                errs = [
                    abs(self._joint_pos.get(j, float('inf')) - float(t))
                    for j, t in zip(UR5E_JOINTS, target_joints)
                ]
                max_err = max(errs)
                if max_err < tolerance:
                    return True
                if max_err < min_err_seen - 0.005:
                    min_err_seen = max_err
                    last_improvement = time.time()
                elif time.time() - last_improvement > STALL_TIMEOUT:
                    self.get_logger().warn(
                        f"Arm stalled — max_err={max_err:.3f} rad, "
                        f"no improvement in {STALL_TIMEOUT:.0f}s"
                    )
                    return False
            time.sleep(0.1)

        self.get_logger().error(
            f"Arm move timed out ({wall_timeout:.0f}s wall, {sim_duration:.1f}s sim) "
            f"— max joint error = {max_err:.3f} rad"
        )
        return False

    def _on_piece_positions(self, msg: String):
        """Update latest camera detections from piece_detector."""
        try:
            self._detected_pieces = json.loads(msg.data)
        except Exception:
            pass

    def _find_detected_piece(self, color: str, expected_xy: tuple, max_dist: float = 0.08):
        """Return the detected piece XY closest to expected_xy, or None if none within max_dist."""
        candidates = self._detected_pieces.get(color, [])
        if not candidates:
            return None
        ex, ey = expected_xy
        best = min(candidates, key=lambda p: (p[0] - ex) ** 2 + (p[1] - ey) ** 2)
        dist = math.hypot(best[0] - ex, best[1] - ey)
        if dist > max_dist:
            self.get_logger().warn(
                f'Closest {color} detection ({best[0]:.3f},{best[1]:.3f}) is {dist*1000:.0f}mm '
                f'from expected ({ex:.3f},{ey:.3f}) — beyond {max_dist*1000:.0f}mm threshold'
            )
            return None
        self.get_logger().info(
            f'Camera-corrected {color} target: ({best[0]:.3f},{best[1]:.3f}) '
            f'(was ({ex:.3f},{ey:.3f}), delta {dist*1000:.0f}mm)'
        )
        return best

    def _compute_pick_joints(self, px: float, py: float, seed_joints: list):
        """Compute approach, mid, and grasp joint configs for the given XY position.

        Returns (approach_j, mid_j, grasp_j) or None if IK fails at approach height.
        """
        approach_raw = self._call_ik(self._ik_cli, px, py, PICK_APPROACH_Z, seed_joints)
        if approach_raw is None:
            return None
        approach_j = self._normalize_joints(approach_raw, seed_joints)

        mid_raw = self._call_ik(self._ik_cli, px, py, PICK_MID_Z, approach_j,
                                avoid_collisions=False)
        mid_j = self._normalize_joints(mid_raw, approach_j) if mid_raw else approach_j

        grasp_raw = self._call_ik(self._ik_cli, px, py, PICK_GRASP_Z, mid_j,
                                  avoid_collisions=False)
        grasp_j = self._normalize_joints(grasp_raw, mid_j) if grasp_raw else mid_j

        return approach_j, mid_j, grasp_j

    def _check_grasp(self) -> bool:
        """Return True if gripper is holding a piece (didn't reach commanded close position).

        A successful grasp stops the gripper short of GRIPPER_PIECE because the
        piece's body blocks full closure.  If the gripper reached the commanded
        position, it closed in air and the pick failed.
        """
        time.sleep(0.4)  # let gripper settle after close command
        actual = self._joint_pos.get('robotiq_85_left_knuckle_joint')
        if actual is None:
            self.get_logger().warn('No gripper joint state — assuming grasp succeeded')
            return True
        grasped = actual < (GRIPPER_PIECE - 0.04)
        self.get_logger().info(
            f'Gripper check: actual={actual:.3f} rad, target={GRIPPER_PIECE:.3f} rad '
            f'→ {"GRASPED" if grasped else "MISSED (air close)"}'
        )
        return grasped

    def _on_pick_and_place(self, msg: String):
        """Handle /connect_four/pick_and_place messages, format: 'color,col'."""
        parts = msg.data.strip().split(',')
        if len(parts) != 2:
            self.get_logger().error(
                f"Invalid pick_and_place message: '{msg.data}' — expected 'color,col'"
            )
            return

        color = parts[0].strip().lower()
        if color not in ('red', 'yellow'):
            self.get_logger().error(f"Unknown color '{color}' — must be 'red' or 'yellow'")
            return

        try:
            col = int(parts[1].strip())
        except ValueError:
            self.get_logger().error(f"Invalid column '{parts[1]}' — must be integer 0-6")
            return

        if not 0 <= col <= 6:
            self.get_logger().error(f"Column {col} out of range — must be 0-6")
            return

        if self._busy:
            self.get_logger().warn(
                f"Arm busy — ignoring pick_and_place '{msg.data}'"
            )
            return

        threading.Thread(
            target=self._execute_pick_and_place,
            args=(color, col),
            daemon=True,
        ).start()

    def _execute_pick_and_place(self, color: str, col: int):
        """Full pick-place-release sequence with camera targeting and grasp verification."""
        with self._lock:
            self._busy = True
            try:
                idx = self._next_piece[color]
                if idx >= len(TRAY_PIECES[color]):
                    self.get_logger().error(f"No more {color} pieces in tray (used all 7)")
                    return

                key = f"{color}_{idx}"
                if key not in self._tray_joints:
                    self.get_logger().error(
                        f"No precomputed IK for tray position {key} — aborting"
                    )
                    return

                nominal_xy = TRAY_PIECES[color][idx]
                self.get_logger().info(
                    f"Pick-place: {color} piece #{idx} from "
                    f"({nominal_xy[0]:.3f}, {nominal_xy[1]:.3f}) → column {col}"
                )

                # Step 1: open gripper
                self.get_logger().info("Step 1: Opening gripper")
                self._move_gripper(GRIPPER_OPEN)

                # Step 2: move to approach height above nominal piece position
                self.get_logger().info(f"Step 2: Moving to tray approach ({key})")
                self._move_arm_joints(self._tray_joints[key], sim_duration=8.0)

                # ── Pick attempt loop (up to MAX_PICK_RETRIES+1 total tries) ──
                grasped = False
                approach_j = self._tray_joints[key]
                mid_j      = self._tray_mid_joints.get(key, approach_j)
                grasp_j    = self._tray_grasp_joints.get(key, mid_j)

                for attempt in range(MAX_PICK_RETRIES + 1):
                    if attempt > 0:
                        self.get_logger().info(
                            f"Retry {attempt}/{MAX_PICK_RETRIES}: re-detecting piece position"
                        )
                        self._move_gripper(GRIPPER_OPEN)
                        # Lift to approach before re-targeting
                        self._move_arm_joints(approach_j, sim_duration=4.0)

                    # Camera correction: replace nominal target with detected position
                    target_xy = nominal_xy
                    detected = self._find_detected_piece(color, nominal_xy)
                    if detected is not None:
                        target_xy = detected
                        # Recompute IK for the actual piece location
                        result = self._compute_pick_joints(
                            target_xy[0], target_xy[1], approach_j
                        )
                        if result is not None:
                            approach_j, mid_j, grasp_j = result
                        else:
                            self.get_logger().warn(
                                "IK failed for detected position — using precomputed joints"
                            )
                    else:
                        self.get_logger().info(
                            "No camera detection available — using precomputed joints"
                        )

                    # Step 3a: descend to mid-height (large motion, low XY drift risk)
                    self.get_logger().info(f"Step 3a (attempt {attempt+1}): Descend to mid")
                    self._move_arm_joints(mid_j, sim_duration=4.0, tolerance=0.05)

                    # Step 3b: final descent to grasp height (short, accurate)
                    self.get_logger().info(f"Step 3b (attempt {attempt+1}): Descend to grasp")
                    self._move_arm_joints(grasp_j, sim_duration=3.0, tolerance=0.05,
                                         wall_timeout=30.0)

                    # Step 4: close gripper and check
                    self.get_logger().info(f"Step 4 (attempt {attempt+1}): Closing gripper")
                    self._move_gripper(GRIPPER_PIECE)
                    grasped = self._check_grasp()

                    if grasped:
                        break
                    self.get_logger().warn(
                        f"Grasp attempt {attempt+1} failed — "
                        f"{'retrying' if attempt < MAX_PICK_RETRIES else 'giving up'}"
                    )

                if not grasped:
                    self.get_logger().error(
                        f"All {MAX_PICK_RETRIES+1} grasp attempts failed — aborting"
                    )
                    self._move_gripper(GRIPPER_OPEN)
                    self._move_arm_joints(HOME_JOINTS, sim_duration=8.0)
                    return

                # Step 5: two-stage lift
                self.get_logger().info("Step 5a: Lifting to mid")
                self._move_arm_joints(mid_j, sim_duration=3.0, tolerance=0.05)
                self.get_logger().info("Step 5b: Lifting to approach height")
                self._move_arm_joints(approach_j, sim_duration=4.0)

                # Step 6: move to column drop position
                self.get_logger().info(f"Step 6: Moving to column {col}")
                if col in self._column_joints:
                    self._move_arm_joints(self._column_joints[col], sim_duration=8.0)
                else:
                    cx, cy, cz = COLUMN_POSES[col]
                    self._moveit2.move_to_pose(
                        position=[cx, cy, cz], quat_xyzw=QUAT_DOWN, cartesian=False
                    )
                    self._wait_executed()

                # Step 7: release
                self.get_logger().info("Step 7: Releasing piece")
                self._move_gripper(GRIPPER_OPEN)
                time.sleep(0.5)

                # Step 8: return home
                self.get_logger().info("Step 8: Returning home")
                self._move_arm_joints(HOME_JOINTS, sim_duration=8.0)

                self._next_piece[color] += 1
                self.get_logger().info(
                    f"Pick-place complete: {color} → col {col} "
                    f"({self._next_piece[color]}/7 {color} pieces used)"
                )

            except Exception as exc:
                self.get_logger().error(f"Pick-place failed: {exc}")
            finally:
                self._busy = False

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
                self._move_arm_joints(HOME_JOINTS, sim_duration=8.0)
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
                    self._move_arm_joints(self._column_joints[col], sim_duration=8.0)
                else:
                    self.get_logger().warn(
                        f"No precomputed IK for col {col} — using MoveIt pose planning"
                    )
                    self._moveit2.move_to_pose(
                        position=[x, y, z],
                        quat_xyzw=QUAT_DOWN,
                        cartesian=False,
                    )
                    self._wait_executed()

                self.get_logger().info(f"Column {col} reached — returning home")
                self._move_arm_joints(HOME_JOINTS, sim_duration=8.0)
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
