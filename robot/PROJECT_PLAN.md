# Connect Four Robot Arm — Full Project Plan

## Project Overview

Build a robot arm that can play Connect Four against a human opponent by:

- Viewing a standard store-bought Connect Four board with a depth camera
- Detecting the board state and determining whose turn it is
- Calling into your existing AlphaZero AI model to choose the best move
- Physically picking up a piece and dropping it into the correct column
- Waiting for the human to play, then repeating

Your stack: Apple Silicon Mac (development) → Ubuntu 24.04 runtime machine → ROS2 Jazzy + Gazebo Harmonic + MoveIt2 + Python + ONNX → real arm (TBD)

---

## Robot Arm Recommendations

### Budget Tier (~$200–$400): Hiwonder ArmPi Ultra (Raspberry Pi version)

- Price: ~$300–$400 depending on configuration
- Specs: 6-DOF, bus servos (25 KG torque), 3D depth camera included, ROS2 compatible
- Pros: Comes with a depth camera out of the box, extensive tutorials, Python/OpenCV support, ROS2 ready.
- Cons: Raspberry Pi based (not Jetson), lower torque. Smaller community than Elephant Robotics.
- Best for: Getting started quickly at low cost.

### Mid Tier (~$500–$800): Elephant Robotics myCobot 280

- Price: ~$650–$800 depending on version
- Specs: 6-DOF, 250g payload, 280mm reach, ±0.5mm repeatability
- Pros: Largest community of any hobby arm, excellent ROS2 + MoveIt support. Huge ecosystem of accessories.
- Cons: 250g payload is light. No depth camera included.
- Best for: Best overall balance of community support, documentation, and ROS2 integration.

### High Tier (~$700–$1,100+): Hiwonder JetArm (Jetson Orin Nano version)

- Price: ~$700–$1,100+ (includes Jetson Orin Nano board)
- Specs: 6-DOF, bus servos (35 KG torque), 3D depth camera (Gemini) included, 6-mic array, ROS2 + MoveIt
- Pros: Natively designed for Jetson Orin Nano. Includes 3D depth camera. Working examples for 3D spatial grasping. Full IK source code and Gazebo simulation included.
- Cons: Higher price. More reliant on Hiwonder's own tutorials/packages.
- Best for: Most turnkey "Jetson + arm + depth camera + ROS2" experience.

### Honorable Mention: Seeed Studio reBot Arm B601

- Price: Target sub-$1,000 build cost (open source, you source parts)
- Specs: 6-DOF + gripper, 650mm reach, 1.5kg payload

**Recommendation:** Go with the Hiwonder JetArm (Orin Nano 8GB version) for the most integrated experience, or the Elephant Robotics myCobot 280 for the best community/ROS2 support.

---

## End Effector: Robotiq 2F-85 Parallel Gripper

**Recommendation: Robotiq 2F-85** (or a compatible open-source clone for cheaper arms).

- **Why fingers over suction:** Fingers are far more general-purpose, required for picking up pieces stacked vertically in a supply tray, and better suited for VLA training data diversity.
- **Why Robotiq 2F-85 specifically:**
  - Industry standard — massive ROS2/Gazebo ecosystem, well-maintained URDF and MoveIt configs
  - 85mm open width is ideal for Connect Four discs (~33mm diameter) — plenty of clearance
  - Adaptive fingers conform to object shape for reliable grasp
  - Excellent sim-to-real transfer: the Gazebo model behaves like the real hardware
  - Compatible with UR5e (which is what the simulation uses)
- **For cheap real arm:** An open-source Robotiq-inspired 2-finger gripper (~$50–200 for servo-driven clone) or the myCobot's included gripper accessory. The VLA model trained on Robotiq 2F-85 in sim should transfer reasonably well to any parallel gripper with similar geometry.
- **Connect Four piece grip strategy:** Approach from above with fingers spread slightly wider than 33mm, lower onto the disc (pieces stored flat in a tray), close fingers to ~25mm to grip the disc by its edge.

---

## Phased Project Plan

### Phase 1: Vision Pipeline on Mac (No ROS2, No Arm) ✅ COMPLETE

Goal: Build and test the computer vision system that can look at a Connect Four board image and extract the full board state.

#### Deliverables ✅
- `board_detector.py` — OpenCV board state extraction
- `turn_tracker.py` — Turn detection and game state management
- `ai_player.py` — ONNX model inference wrapper (AlphaZero, 3-channel input)
- `test_pipeline.py` — End-to-end test with sample images (8/8 perfect accuracy)
- `test_images/` — Synthetic test images with ground truth
- `requirements.txt` — Python dependencies

---

### Phase 2: Live Camera Feed on Mac 🔄 MOSTLY COMPLETE

Goal: Replace static images with a live camera feed and validate real-time board detection.

#### 2.1 — Live Camera Capture ✅
- ✅ OpenCV VideoCapture streaming, camera calibration, adaptive HSV baseline sampling at lock time, frame rate throttling

#### 2.2 — Robust Detection Tuning ✅
- ✅ Adaptive thresholding, temporal smoother, burst guard, gravity-validity constraint
- ✅ `StableStateDetector` (configurable via `--stable-seconds`)
- ✅ Shirt/face contamination guards, gravity-strict cascade prevention

#### 2.3 — Game Loop ✅
- ✅ Full cooperative game loop (`game_loop.py`) with AI inference, overlay, TTS, logging

#### 2.4 — YOLO-Enhanced Piece Detection ✅
- ✅ Fine-tuned YOLOv8, `YOLOEnhancedBoardDetector`, hybrid YOLO+HSV fusion

#### 2.5 — Board Overlay Accuracy ✅
- ✅ Extended hole search, clipped-contour guard, accurate board overlay

#### 2.6 — Jetson Orin Nano Deployment ⏸ DEPRIORITIZED
*Deferred until after VLA model is working in simulation (Phase 3C). The vision pipeline already works well on Ubuntu; Jetson deployment is an optimization, not a blocker.*

- [ ] Convert YOLO `.pt` to TensorRT `.engine`
- [ ] Install Jetson-specific `onnxruntime-gpu` wheel
- [ ] Replace macOS TTS with Linux alternative (`espeak-ng` or `pyttsx3`)
- [ ] Validate Orbbec depth camera via Linux/Jetson SDK

#### 2.7 — Depth Camera Integration ⏸ DEPRIORITIZED
*Deferred until after VLA model is working. The VLA model will use wrist/scene cameras for grasping; the Orbbec is primarily for board-state detection which works well without depth already.*

- [ ] Sample board-plane depth at lock time
- [ ] Compute `board_near_mm` / `board_far_mm` range
- [ ] Reject YOLO detections outside board depth range

#### 2.8 — Future Perception Improvements (Backlog)
- [ ] Fine-tune YOLO on captured game-session data
- [ ] Lightweight CNN cell classifier as YOLO replacement

---

### Phase 3: ROS2 Simulation — Core Infrastructure ✅ COMPLETE

Goal: Working Gazebo + MoveIt2 simulation with the arm reaching all 7 columns reliably.

#### Status: Complete as of 2026-04-16

- ✅ Ubuntu 24.04 machine running ROS2 Jazzy
- ✅ Package `connect_four_arm` builds and launches via `connect_four.launch.py`
- ✅ Full stack: Gazebo Harmonic (headless) + ros2_control + MoveIt2 + pymoveit2 client
- ✅ Arm reaches all 7 columns deterministically (IK precomputed at startup, joint-space planning)
- ✅ Visual board/column markers in RViz2
- ✅ Home → column → home sequence working
- ✅ `game_loop.py` ROS integration (`--ros` flag publishes to `/connect_four/drop_column`)
- ✅ `HOME_JOINTS = [0, -π/2, 0, -π/2, 0, 0]`; arm homes on startup and after each drop

#### Key files
```
ros2_ws/src/connect_four_arm/
  launch/
    connect_four.launch.py         # Single-command full stack launch
    connect_four_sim.launch.py     # Gazebo + Xvfb
    connect_four_moveit.launch.py  # MoveIt2 + RViz2
  scripts/
    column_mover.py                # Arm control node (IK precompute + joint-space planning)
  config/
    connect_four_moveit_controllers.yaml
```

#### Board & Column Geometry
```
BOARD_X        = 0.65 m
BOARD_WIDTH    = 0.292 m  (7 cols × 42 mm)
BOARD_HEIGHT   = 0.254 m  (6 rows × 42 mm)
DROP_Z         = 0.304 m  (50 mm above board top)
Column y positions: (3 - i) * COL_SPACING, i = 0..6
  col 0 (left):   y = +0.125 m
  col 3 (center): y =  0.000 m
  col 6 (right):  y = -0.125 m
```

---

### Phase 3B: VLA-Ready Simulation 🔄 IN PROGRESS

Goal: Upgrade the Gazebo scene to be physically accurate and capable of collecting VLA training demonstrations. Success = arm can execute a scripted pick-place-release sequence in sim with realistic physics.

#### 3B.1 — Realistic Gazebo Scene ✅
- ✅ Table at z=0, board (7 column dividers + front/back plates) at (0.65, 0, 0)
- ✅ Pieces: Hasbro dimensions, 38mm dia × 7mm thick, flat in supply tray
  - Red: x=0.50, Yellow: x=0.42, y=-0.126..+0.126 step 0.042m, z=0.0035m
- ✅ Coin-slot guides on board: two rails at top creating 9mm-wide slot (x_world=0.628–0.637)
  - Pieces enter edge-on after wrist reorientation; constrained in X, correct column via Y

#### 3B.2 — Robotiq 2F-85 Gripper Integration ✅
- ✅ `ros-jazzy-robotiq-description` installed; URDF attached to UR5e tool0
- ✅ Mimic joints: ONLY `<param name="mimic">` + `<param name="multiplier">` — no interfaces on mimic joints
- ✅ Gripper controller: JTC on `robotiq_85_left_knuckle_joint` (0.0=open, 0.8=closed)
- ✅ SRDF: 88 disable_collisions pairs for all gripper links (prevents CheckStartStateCollision abort)

#### 3B.3 — Full Pick-Place Sequence ✅ (first successful run achieved)
- ✅ Topic `/connect_four/pick_and_place` (String, format `"color,col"`)
- ✅ Column IK precomputed with `wrist_1 += π/2` — piece arrives edge-on at slot
- ✅ Gripper settle delay (0.35s) before close; slower final descent (5s) prevents early fire
- ✅ PICK_GRASP_Z=0.112m, DROP_Z=0.382m, SLOT_CENTER_X=0.6325m, GRIPPER_PIECE=0.62 rad
- 🔄 Tuning needed: verify wrist_1 is correct rotation joint; tune GRIPPER_PIECE for 38mm piece

#### 3B.4 — Scene Camera Setup ✅ (overhead RGBD mounted)
- ✅ Overhead camera at (0.55, 0, 1.2), pitched down, publishes `/overhead_camera/*`
- [ ] Wrist camera on UR5e tool0 link (optional for SO-101 path — SO-101 has its own wrist cam)

---

### Phase 3C: SO-101 Hardware Setup 🔲 NEXT AFTER 3B

Goal: Order and assemble the LeRobot SO-101 leader+follower arm pair. This replaces the "VLA training in simulation" path — real demonstrations from a human operator are faster and more reliable than sim-generated data.

**Why SO-101 over a UR5e real arm:**
The SO-101 is a ~$150–300 6-DOF open-source arm designed specifically for LeRobot teleoperation and imitation learning. The leader arm lets you physically demonstrate pick-and-place, the follower records it. 50–100 episodes → trainable ACT policy in ~4 hours of GPU time. No sim-to-real gap for the learned behaviors.

#### 3C.1 — What to Order

**Option A — Buy assembled (easiest):**
- Search: "SO-ARM100" or "SO-101" on Seeed Studio, WAVESHARE, or AliExpress
- Buy 2 units (one leader, one follower) — ~$150–250 each assembled
- Also buy: 2× USB webcam (Logitech C270 or similar, ~$25 each) for wrist cameras

**Option B — Buy parts and print (cheapest, ~$100–150/arm):**
- Servos per arm: 5× Feetech STS3215 (~$14 each) + 2× Feetech STS3032 (~$10 each) = ~$90/arm
- USB adapter: 1× Feetech SCServo USB-to-serial board (~$15)
- 3D printed frame: download STLs from `TheRobotStudio/SO-ARM100` on GitHub, print in PLA (~300g per arm)
- Hardware: M2/M3 screws, bearings, power supply (7.4V 2A LiPo or DC adapter)
- Wrist camera: 1× USB webcam per follower arm

**Total for 2 arms (Option B):** ~$250–350 + print costs

**Before ordering, confirm:** your Ubuntu machine has 2 free USB ports (one per arm's controller board) and you have access to a 3D printer or print service.

#### 3C.2 — LeRobot Environment Setup

```bash
# Install LeRobot (run on Ubuntu machine)
git clone https://github.com/huggingface/lerobot.git
cd lerobot
pip install -e ".[feetech]"

# Verify SO-101 is recognized
python lerobot/scripts/find_motors_bus_port.py

# Calibrate both arms (follow the interactive prompts)
python lerobot/scripts/control_robot.py calibrate \
    --robot-path lerobot/configs/robot/so101.yaml \
    --robot-overrides '~motors' --arms main_follower
python lerobot/scripts/control_robot.py calibrate \
    --robot-path lerobot/configs/robot/so101.yaml \
    --robot-overrides '~motors' --arms main_leader
```

#### 3C.3 — Collect Demonstrations

Set up physically:
- Mount follower arm on table, position Connect Four board at comfortable reach (~300mm center from arm base)
- Place piece supply tray with red and yellow pieces within reach
- Attach wrist camera to follower arm's tool link, USB to PC

```bash
# Record 50+ episodes — you control leader arm, follower records
python lerobot/scripts/control_robot.py record \
    --robot-path lerobot/configs/robot/so101.yaml \
    --fps 30 \
    --repo-id <your-hf-username>/connect_four_pick_place \
    --tags connect_four \
    --warmup-time-s 5 \
    --episode-time-s 30 \
    --reset-time-s 10 \
    --num-episodes 60
```

Each episode: pick one piece from tray, drop in a column (vary colors and columns). Reset piece to tray between episodes. Aim for ~10 demos per column.

#### 3C.4 — Train ACT Policy

Requires a machine with GPU (RTX 3080+ recommended; 8GB VRAM minimum).

```bash
# Train ACT on collected demos (~3–5 hours on RTX 3080)
python lerobot/scripts/train.py \
    policy=act_so101_real \
    env=so101_real \
    dataset_repo_id=<your-hf-username>/connect_four_pick_place \
    hydra.run.dir=outputs/train/act_connect_four

# Evaluate the trained policy
python lerobot/scripts/control_robot.py record \
    --robot-path lerobot/configs/robot/so101.yaml \
    -p outputs/train/act_connect_four/checkpoints/last/pretrained_model \
    --fps 30 \
    --repo-id <your-hf-username>/eval_connect_four \
    --num-episodes 20
```

#### 3C.5 — Integration with Game Loop

```bash
# Wire game_loop.py → SO-101 policy instead of scripted MoveIt
# The policy takes: wrist camera image + (optionally) language goal
# Publishes: joint positions to SO-101 follower arm
```

Build a thin ROS2 bridge node: subscribes to `/connect_four/ai_move` (column number), maps to language goal "pick red piece, drop column N", runs ACT inference, streams joint targets to SO-101.

#### Tips for Good Demonstrations
- Be deliberate and smooth — jerky motions transfer poorly
- Pick piece from the same tray position each episode until that slot is well-covered
- Vary drop column across episodes (don't always drop in col 3)
- If a pick fails during recording, stop, reset, and redo that episode
- 50 good demos beat 200 sloppy ones

---

### Phase 4: Full Game Integration 🔲

Goal: SO-101 plays complete Connect Four games against a human.

#### 4.1 — End-to-End Game Test
- [ ] Wire `game_loop.py --ros` → game_manager_node → SO-101 policy node
- [ ] Camera (overhead or wrist) detects board state after each move
- [ ] AI chooses column → policy picks piece + drops it
- [ ] Handle: piece supply running out, failed pick (policy retry), board shift detection

#### 4.2 — Robustness Improvements
- [ ] Collect additional demonstrations for failure cases seen in 4.1
- [ ] Fine-tune policy on augmented dataset
- [ ] Add piece-detection fallback: if pick fails (camera confirms piece didn't move), re-attempt

#### 4.3 — Full Demo Video
- [ ] Record complete game: setup → 7+ moves → win
- [ ] Document: policy checkpoint, dataset, hardware config

---

### Phase 5: Vision Integration for Board Detection 🔲 (DEPRIORITIZED)

*The board detection pipeline (Phase 2) already works well. This phase is only needed for full autonomy — the SO-101 policy handles grasping, but the overhead camera still needs ROS2 integration for board-state publishing.*

#### 5.1 — ROS2 Node Architecture

| Node | Subscribes To | Publishes | Purpose |
|------|--------------|-----------|---------|
| `camera_node` | — | `/camera/image_raw` | USB/depth camera capture |
| `board_detector_node` | `/camera/image_raw` | `/connect_four/board_state` | Board state from image |
| `game_manager_node` | `/connect_four/board_state` | `/connect_four/ai_move` | Turn tracking + AI call |
| `so101_policy_node` | `/connect_four/ai_move` | SO-101 joint commands | ACT policy inference |

#### 5.2 — Jetson Orin Nano Deployment (optional)
- [ ] Deploy ACT policy on Jetson for untethered operation
- [ ] Convert to TensorRT if inference is too slow (target: <100ms per step)

---

### Phase 5: Vision Integration for Board Detection 🔲

*This phase was previously higher priority. It is now sequenced after the VLA model is working because the board detection pipeline (Phase 2) already works well via the game_loop's camera feed, and the Orbbec/ROS integration is only needed for fully autonomous operation.*

#### 5.1 — ROS2 Node Architecture
Create ROS2 nodes to replace the standalone `game_loop.py`:

| Node | Subscribes To | Publishes | Purpose |
|------|--------------|-----------|---------|
| `camera_node` | — | `/camera/image_raw`, `/camera/depth` | Captures from depth camera |
| `board_detector_node` | `/camera/image_raw` | `/connect_four/board_state` | Detects board state from image |
| `game_manager_node` | `/connect_four/board_state` | `/connect_four/ai_move` | Tracks turns, calls AI |
| `arm_controller_node` | `/connect_four/ai_move` | arm commands | Triggers VLA pick-place |

#### 5.2 — Jetson Orin Nano Deployment
- [ ] Convert YOLO `.pt` to TensorRT `.engine`
- [ ] Install `onnxruntime-gpu` for Jetson
- [ ] Replace macOS TTS with `espeak-ng` / `pyttsx3`
- [ ] Validate Orbbec depth camera on Linux/Jetson SDK
- [ ] Deploy full stack on Jetson; profile with `tegrastats`

#### 5.3 — Orbbec Depth Camera Integration
- [ ] Sample board-plane depth at lock time
- [ ] Reject YOLO detections outside `board_near_mm` / `board_far_mm` range
- [ ] Full autonomous loop: no human needed to trigger AI move

---

### Phase 6: Polish + Advanced Features (Optional)

#### 6.1 — Natural Language Commands
- Whisper on Orin Nano (TensorRT) for speech-to-text
- LLM intent parsing: "Play in the middle" → column 3
- Visual grounding for free-form pickup commands

#### 6.2 — Other Enhancements
- Voice feedback: "Your turn!" / "I'm thinking..." / "I win!"
- LED column indicators
- Automatic board reset (arm pulls release slider)
- Difficulty levels (vary MCTS simulation count)
- Web dashboard: stream camera feed + game state
- Multiple game support (tic-tac-toe, checkers)

---

## Summary Timeline

| Phase | Status | Description |
|-------|--------|-------------|
| Phase 1: Vision on Mac (static images) | ✅ COMPLETE | Board detection, AI inference |
| Phase 2: Live camera on Mac | ✅ MOSTLY COMPLETE | Game loop, YOLO, robust detection |
| Phase 3: ROS2 simulation — core | ✅ COMPLETE | UR5e + Gazebo + MoveIt2 + column moves |
| Phase 3B: VLA-ready simulation | 🔄 IN PROGRESS | Pieces + gripper + pick-place — tuning |
| Phase 3C: SO-101 hardware setup | 🔲 NEXT | Order parts, assemble, calibrate, collect demos, train ACT |
| Phase 4: Full game integration | 🔲 | SO-101 plays complete games vs. human |
| Phase 5: Vision / ROS2 integration | 🔲 DEPRIORITIZED | ROS2 board detection pipeline |
| Phase 6: Polish | 🔲 | NLP, voice, web dashboard |

**Current focus: Finish Phase 3B tuning** (wrist rotation direction, GRIPPER_PIECE value), then **order SO-101 parts for Phase 3C**.

**When to order SO-101:** Now or any time during 3B tuning. Parts take 1–3 weeks to arrive (longer for Option B printed parts). By the time you finish 3B and print/assemble the arms, you'll be ready to collect demonstrations immediately.

**Hardware path summary:** SO-101 leader+follower → LeRobot ACT policy → trained on real demos → plays Connect Four. The UR5e simulation validated the algorithm; the SO-101 is the real-hardware endpoint at a fraction of the cost.
