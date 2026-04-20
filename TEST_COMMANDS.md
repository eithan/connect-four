# Connect Four Robot Arm — Test Commands

## Prerequisites

Build the workspace after any code changes:
```bash
cd ~/development/connect-four/ros2_ws
source /opt/ros/jazzy/setup.bash
colcon build --packages-select connect_four_arm
```

Source the workspace before running any command:
```bash
source ~/development/connect-four/ros2_ws/install/setup.bash
```

---

## 1. Launch the Full Simulation

Starts Gazebo (headless), MoveIt2, RViz2, and column_mover in sequence.
Expected: takes ~30 seconds to fully initialize; you'll see log messages
"MoveIt2 ready", "IK col 0..6 precomputed", "Tray IK precomputed: 14/14",
"Moving to home position", "Home position reached".

```bash
ros2 launch connect_four_arm connect_four.launch.py
```

Watch for these success indicators in the logs:
- `[column_mover]: MoveIt2 ready — precomputing column IK`
- `[column_mover]: Column IK precomputed: 7/7 columns`
- `[column_mover]: Tray IK precomputed: 14/14 positions (approach + grasp)`
- `[column_mover]: Home position reached`

If IK fails for some columns: the arm will fall back to MoveIt pose planning for those columns (slower but should still work).

---

## 2. Pick-and-Place Sequence (Primary Test)

After the simulation is fully initialized (home position reached):

### Single red piece → column 3 (center)
```bash
ros2 topic pub --once /connect_four/pick_and_place std_msgs/String "{data: 'red,3'}"
```
**Expected behavior (8 steps):**
1. Gripper opens (robotiq_85_left_knuckle_joint → 0.0 rad)
2. Arm moves to tray approach above red piece #0 (x=0.50, y=-0.126, z≈0.169) — ~30 wall-sec
3. Arm descends to grasp height (z≈0.114) — ~15 wall-sec
4. Gripper closes (0.55 rad, ~33mm gap)
5. Arm lifts back to approach height — ~15 wall-sec
6. Arm moves to column 3 (x=0.65, y=0.0, z≈0.413) — ~30 wall-sec
7. Gripper opens, piece falls into column 3
8. Arm returns to home — ~30 wall-sec

**Total wall time: ~2–5 minutes** (depends on RTF; sim runs at ~25–30% RTF)

**Log messages to watch for:**
```
[column_mover]: Pick-place: red piece #0 from (0.500, -0.126) → column 3
[column_mover]: Step 1: Opening gripper
[column_mover]: Step 2: Moving to tray approach (red_0)
[column_mover]: Step 2 done: wall=Xs sim=Ys RTF=Z%
[column_mover]: Step 3: Descending to grasp height
[column_mover]: Step 4: Closing gripper
[column_mover]: Step 5: Lifting to approach height
[column_mover]: Step 6: Moving to column 3
[column_mover]: Step 7: Releasing piece
[column_mover]: Step 8: Returning home
[column_mover]: Pick-place complete: red → col 3 (1/7 red pieces used)
```

### Single yellow piece → column 0
```bash
ros2 topic pub --once /connect_four/pick_and_place std_msgs/String "{data: 'yellow,0'}"
```

### Multiple pieces (run sequentially — wait for previous to finish)
```bash
ros2 topic pub --once /connect_four/pick_and_place std_msgs/String "{data: 'red,3'}"
# wait for "Pick-place complete" in logs, then:
ros2 topic pub --once /connect_four/pick_and_place std_msgs/String "{data: 'yellow,3'}"
# wait, then:
ros2 topic pub --once /connect_four/pick_and_place std_msgs/String "{data: 'red,4'}"
```

---

## 3. Simple Column Drop Test (No Gripper, Legacy)

Tests arm movement only (no pick-place, no gripper). Arm moves above a column then returns home.

```bash
ros2 topic pub --once /connect_four/drop_column std_msgs/Int32 "{data: 3}"
```
**Expected:** Arm moves above column 3, then returns home. Takes ~1–2 minutes wall time.

---

## 4. Monitor Simulation Status

### Check arm joint positions in real time
```bash
ros2 topic echo /joint_states --once
```
At home position, expect approximately:
- shoulder_pan_joint: 0.0
- shoulder_lift_joint: -1.571
- elbow_joint: 0.0
- wrist_1_joint: -1.571
- wrist_2_joint: 0.0
- wrist_3_joint: 0.0
- robotiq_85_left_knuckle_joint: 0.0 (open)

### Check active controllers
```bash
ros2 control list_controllers
```
Expected active: `joint_state_broadcaster`, `joint_trajectory_controller`, `gripper_controller`

### Check controller_manager is running
```bash
ros2 service list | grep controller_manager
```

### Monitor pick-and-place RTF and timing
Watch the column_mover log for `Step 2 done: wall=Xs sim=Ys RTF=Z%`.
- RTF ~25-30% is normal for this simulation
- wall time per large move should be 20–40 seconds with the new position_proportional_gain=2.0

### Check the gz_ros_control gain was applied
```bash
ros2 param get /gz_ros_control position_proportional_gain
```
Expected: `2.0`

---

## 5. Troubleshooting

### If arm doesn't move (error: "Arm JTC action server not available")
The JTC isn't running. Check controllers:
```bash
ros2 control list_controllers
```
If `joint_trajectory_controller` is missing, the sim may still be starting up. Wait 15 more seconds.

### If arm times out ("Arm move timed out")
The arm isn't converging to target. Check:
1. The `position_proportional_gain` is 2.0: `ros2 param get /gz_ros_control position_proportional_gain`
2. RTF might be very low — check `Step 2 done:` log line for RTF%
3. Increase `sim_duration` in `column_mover.py` (current: 8.0 for large moves)

### If IK fails ("No precomputed IK for tray position")
Restart the simulation. KDL sometimes fails on first solve after startup.

### If gripper doesn't open/close ("Gripper action server not available")
Check: `ros2 control list_controllers | grep gripper`
The gripper_controller spawns at t=12s, wait a few more seconds.

### Kill and restart cleanly
```bash
pkill -f "ros2 launch" && pkill -f "gz sim" && pkill -f "column_mover" && pkill -f "move_group"
sleep 3
ros2 launch connect_four_arm connect_four.launch.py
```

---

## 6. What Changed (Session Summary)

### Root Cause Fixed
`gz_ros2_control`'s `GazeboSimSystem` converts position commands to velocity using:
```
target_velocity = position_proportional_gain × (cmd − current) × update_rate
```
The default `position_proportional_gain = 0.1` produced only **0.022 rad/sim-sec** convergence for a 2 rad elbow move (~95 sim-seconds to complete a single move = 5+ minutes wall time per move).

### Fix Applied
Added `<position_proportional_gain>2.0</position_proportional_gain>` to the Gazebo plugin SDF element in `urdf/ur5e_with_gripper.urdf.xacro`. This 20× increase should yield ~**0.44 rad/sim-sec** → converges within ~5 sim-seconds (~18 wall-seconds at 27% RTF).

### Other Changes
- `column_mover.py`: increased `sim_duration` from 4.0→8.0 (large moves) and 2.0→4.0 (small moves) to give the JTC trajectory more sim-time, providing more margin at lower RTF values.
- Direct JTC arm control (bypasses MoveIt/Pilz) was already in place from a prior session — this remains unchanged.

### Architecture (for reference)
- `position_proportional_gain` is read from the SDF `<plugin>` element, NOT from the controllers YAML
- It is a SINGLE global gain for ALL joints (arm + gripper) — not per-joint
- The formula uses `update_rate = 250 Hz` (from controller_manager config)
- The gain produces `JointVelocityCmd`, not force/effort
- Source: `/tmp/ros-jazzy-gz-ros2-control-1.2.17/src/gz_ros2_control_plugin.cpp` line 313
