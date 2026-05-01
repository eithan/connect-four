# Connect Four Robot Arm — Full Project Plan

> **Document map.** This is the master plan. Companion docs in this folder:
> - [`SESSION_STATUS.md`](./SESSION_STATUS.md) — concise current state, hardware ordered, immediate next steps. Read this first if returning after a break.
> - [`ARM_DECISION_LOG.md`](./ARM_DECISION_LOG.md) — historical record of the arm-selection process (JetArm evaluated and rejected, SO-101 chosen).
> - [`VLA_FINETUNING_PLAN.md`](./VLA_FINETUNING_PLAN.md) — deep-dive on Vision-Language-Action fine-tuning (SmolVLA / π₀ / π₀.₅) that complements Phase 3C and Phase 4 below.
> - [`CLAUDE.md`](./CLAUDE.md) — guidance for Claude when working in this folder.

## Project Overview

Build a robot arm that can play Connect Four against a human opponent by:

- Viewing a standard store-bought Connect Four board with a depth camera
- Detecting the board state and determining whose turn it is
- Calling into your existing AlphaZero AI model to choose the best move
- Physically picking up a piece and dropping it into the correct column
- Waiting for the human to play, then repeating

Your stack: Apple Silicon Mac (development) → Ubuntu 24.04 runtime machine → ROS2 Jazzy + Gazebo Harmonic + MoveIt2 + Python + ONNX → real arm (TBD)

---

## Robot Arm — Decision

**Locked: LeRobot SO-ARM101** (leader + follower, fully assembled). See [`ARM_DECISION_LOG.md`](./ARM_DECISION_LOG.md) for the full evaluation history including the JetArm evaluation, Hiwonder support replies (429 mm reach, 2–4 mm precision), and the rationale for choosing SO-101 over alternatives.

Summary of why SO-101: ecosystem alignment with HuggingFace LeRobot and Physical Intelligence's openpi (the de facto VLA toolchain), built-in leader/follower teleoperation, cross-embodiment skill transfer, alignment with the existing UR5e/Robotiq 2F-85 ROS2 simulation, and compatibility with the long-term goal of generalist robot teaching for household tasks and other games.

---

## End Effector

**Why fingers over suction:** Fingers are far more general-purpose, required for picking up pieces stacked flat in a supply tray, and better suited for VLA training-data diversity. No suction cups in this project.

The simulation and the real hardware use **different parallel-jaw grippers**. This is fine for our path; details below.

### Simulation gripper: Robotiq 2F-85 URDF (free, in Gazebo)

The existing `connect_four_arm` ROS2 package uses the **Robotiq 2F-85 URDF** mounted on the UR5e in Gazebo. This is a community-maintained open-source URDF + MoveIt config + ROS2 controllers package (`ros-jazzy-robotiq-description`), installed via `apt`. **Not a real-hardware purchase** — the actual Robotiq 2F-85 gripper retails for $5,300–$8,000 and is not needed here.

Why it was chosen for sim:
- Industry-standard ROS2/Gazebo model, well-maintained URDF and MoveIt configs
- 85 mm open width is plenty of clearance for ~38–40 mm Connect Four discs
- Already integrated and working in Phase 3B (mimic joints, SRDF disable_collisions, JTC controller)
- No reason to swap it out

### Real-hardware gripper: SO-101 stock parallel-jaw (included with kit)

The LeRobot **SO-ARM101 ships with its own parallel-jaw gripper** as part of the assembled kit — a Feetech STS3215-driven open-source design that is the canonical end-effector for the LeRobot ecosystem. Every published SmolVLA / π₀ / π₀.₅ recipe targeting SO-100/101 expects exactly this gripper. **Zero incremental cost; nothing additional to buy.**

Connect Four discs (33–40 mm diameter, ~6–7 mm thick) are well within its grasp range.

### Sim-vs-real gripper mismatch — why it doesn't matter

There is a real geometry mismatch between the Robotiq 2F-85 in Gazebo and the SO-101 stock gripper in the real arm. This is acceptable for our path because:

- **Phase 3C uses real teleoperated demonstrations**, not sim-trained policies, for the learned controller. The VLA model is fine-tuned on real-arm data, so the sim-side gripper geometry doesn't affect the deployed policy.
- **The Gazebo simulation continues to validate higher-level algorithms** — game state, column targeting, MoveIt2 collision avoidance, end-to-end ROS2 plumbing. All gripper-agnostic.
- **If we ever wanted sim-trained policies that transfer to real** (sim-to-real reinforcement learning, for example), we would swap the Gazebo URDF to the community SO-101 URDF. That's a separate project, not on the critical path for v1.

### Connect Four piece grip strategy (applies to both sim and real)

Approach from above with fingers spread slightly wider than the disc OD (~40 mm), lower onto the disc (pieces stored flat in a tray), close fingers until contact is detected to grip by the rim. Custom 3D-printed TPU fingertips with a concave cradle for the disc edge are an optional Phase 3C iteration if grip reliability falls short with stock fingers.

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
*Deferred to Phase 5 (untethered/edge deployment). The vision pipeline already works well on the existing Ubuntu machine; Jetson deployment is an optimization for standalone operation, not a blocker. See [`SESSION_STATUS.md`](./SESSION_STATUS.md) "Deferred to Phase 5" hardware section for the buy trigger.*

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

### Phase 3B: VLA-Ready Simulation ✅ ALGORITHM-COMPLETE

Goal: Upgrade the Gazebo scene to be physically accurate and capable of collecting VLA training demonstrations. Success = arm can execute a scripted pick-place-release sequence in sim with realistic physics.

**Closeout note (2026-04-30):** Declared algorithm-complete. Geometry, IK, MoveIt2 motion planning, gripper controller, ROS2 plumbing, and full launch graph are validated. Gazebo Harmonic's contact physics for small parallel-jaw grasps proved unreliable, and reliable sim grasping is no longer a blocker because (a) Phase 3C trains on real teleoperated demonstrations, not sim-trained policies, and (b) the gripper geometry mismatch between sim (Robotiq 2F-85) and real (SO-101 stock parallel-jaw) means sim grasp behavior wouldn't transfer anyway. Sub-task 3B.3 grasp tuning is parked.

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

#### 3B.3 — Full Pick-Place Sequence ✅ (motion plumbing validated; grasp contact physics parked)
- ✅ Topic `/connect_four/pick_and_place` (String, format `"color,col"`)
- ✅ Column IK precomputed with `wrist_1 += π/2` — piece arrives edge-on at slot
- ✅ Gripper settle delay (0.35s) before close; slower final descent (5s) prevents early fire
- ✅ PICK_GRASP_Z=0.112m, DROP_Z=0.382m, SLOT_CENTER_X=0.6325m, GRIPPER_PIECE=0.62 rad
- 🅿️ Parked: wrist_1 rotation direction verification + GRIPPER_PIECE tuning. Gazebo grasp contact physics are unreliable for small parallel-jaw grasps; the sim mostly misses pickups. Not a blocker — Phase 3C uses real teleop demos, and the sim/real gripper geometry mismatch means sim grasp behavior wouldn't transfer anyway.

#### 3B.4 — Scene Camera Setup ✅ (overhead RGBD mounted)
- ✅ Overhead camera at (0.55, 0, 1.2), pitched down, publishes `/overhead_camera/*`
- [ ] Wrist camera on UR5e tool0 link (optional for SO-101 path — SO-101 has its own wrist cam)

---

### Architecture Insights from Physical Intelligence (informs Phases 3C and 4)

Physical Intelligence (makers of π₀, π₀.₅, π₀.₇, openpi) has open-sourced their architecture and we adopt several patterns from it. Detail in [`VLA_FINETUNING_PLAN.md`](./VLA_FINETUNING_PLAN.md); the headlines:

1. **Separate policy from execution; stream action chunks.** Large VLAs run on GPU servers and stream chunks of ~50 actions to the robot over websocket. Our ROS2 action contract should support this directly.
2. **LeRobot dataset format is the de facto interchange standard.** All published VLA fine-tuning recipes consume it (SmolVLA, π₀, π₀.₅, OpenVLA, ACT, Diffusion Policy). Phase 3C demonstration recording must produce LeRobot-format datasets.
3. **Don't train from scratch — fine-tune a pre-trained foundation VLA.** PI's whole thesis. For Connect Four: fine-tune SmolVLA on hundreds (not thousands) of demos.
4. **Cross-embodiment generalization.** Skills transfer across arm types via the LeRobot toolchain. This is why the existing UR5e simulation continues to add value alongside the SO-101 real-arm work.
5. **For on-Jetson inference, SmolVLA is the ceiling, not π₀ or larger.** SmolVLA (450M) was built specifically as the on-device counterpart to PI's cloud models. π₀ (3B) and π₀.₅ require remote inference from a desktop GPU.

These patterns mostly reinforce decisions already in this plan — keeping the policy decoupled from arm motion (clean ROS2 action interface), targeting LeRobot via the SO-101, planning ACT before VLA. The main concrete additions are: (a) record demos in LeRobot format from day one, (b) add language annotations to episodes during Phase 3C, (c) keep the option open in Phase 4+ to add a `arm_node_openpi` variant that talks to a remote π₀.₅ policy server.

---

### Phase 3C: SO-101 Hardware Setup 🔄 IN FLIGHT (arm ordered, prep work in progress)

Goal: Order, assemble, calibrate, and integrate the LeRobot SO-101 leader+follower arm pair. This replaces the "VLA training in simulation" path — real demonstrations from a human operator are faster and more reliable than sim-generated data.

**See also:** [`VLA_FINETUNING_PLAN.md`](./VLA_FINETUNING_PLAN.md) for the openpi/SmolVLA/π₀/π₀.₅ deep-dive that extends this phase. [`SESSION_STATUS.md`](./SESSION_STATUS.md) has the live TODO list of pre-arm prep work.

**Why SO-101 over a UR5e real arm:**
The SO-101 is an open-source 6-DOF arm designed specifically for LeRobot teleoperation and imitation learning. The leader arm lets you physically demonstrate pick-and-place, the follower records it. 50–100 episodes → trainable ACT policy in ~4 hours of GPU time. No sim-to-real gap for the learned behaviors. See [`ARM_DECISION_LOG.md`](./ARM_DECISION_LOG.md) for the full evaluation that led to this choice (notably over the Hiwonder JetArm).

#### 3C.1 — What Was Ordered ✅

**Hiwonder LeRobot SO-ARM101 Advanced Kit (assembled), ordered 2026-04-30, ETA early June 2026 (~25 business days from China). Total: $540 ($460 + $80 shipping).**

Bundled in the Advanced Kit:
- Leader + follower arms (both fully assembled)
- Wrist camera + external scene camera
- 12 servos total (6 × 12V, 6 × 7.4V) with magnetic feedback
- 2× motor control boards
- BusLinker V3.0 debugging board
- All cables, power supplies (12V + 5V), table clamps

**No additional camera purchase needed** — both wrist and scene cameras ship with the kit.

**Before the arm arrives, confirm on the Ubuntu machine:**
- At least 2 free USB ports for arm controllers (4 total counting cameras)
- Discrete GPU available for ACT training (RTX 3080+ recommended)
- LeRobot environment installed and smoke-tested (TODO item 1 in `SESSION_STATUS.md`)

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

### Phase 5: ROS2 Vision Integration + Jetson Deployment 🔲 DEPRIORITIZED

*The board detection pipeline (Phase 2) already works well as a standalone Python process. This phase is only needed for full autonomy / untethered operation — the SO-101 policy (Phase 3C/4) handles grasping, but the overhead camera still needs ROS2 integration for board-state publishing if you want to remove the standalone `game_loop.py`. The Jetson Orin Nano (deferred from earlier phases — see [`SESSION_STATUS.md`](./SESSION_STATUS.md) "Deferred to Phase 5") is the buy trigger here.*

#### 5.1 — ROS2 Node Architecture

Replace the standalone `game_loop.py` with ROS2 nodes:

| Node | Subscribes To | Publishes | Purpose |
|------|--------------|-----------|---------|
| `camera_node` | — | `/camera/image_raw` (and `/camera/depth` if depth is added) | USB camera capture (SO-101 wrist + scene cameras, or external) |
| `board_detector_node` | `/camera/image_raw` | `/connect_four/board_state` | Board state from image (port of `board_detector.py`) |
| `game_manager_node` | `/connect_four/board_state` | `/connect_four/ai_move` | Turn tracking + AlphaZero call (port of `turn_tracker.py` + `ai_player.py`) |
| `so101_policy_node` (or `arm_node`) | `/connect_four/ai_move` | SO-101 joint commands | ACT/SmolVLA policy inference; thin wrapper over LeRobot Python |

#### 5.2 — Jetson Orin Nano Deployment (optional, untethered operation)

- [ ] Buy Jetson Orin Nano 8GB Super Dev Kit + 256GB NVMe SSD (~$340) — see [`SESSION_STATUS.md`](./SESSION_STATUS.md) for trigger criteria
- [ ] Flash JetPack 6.x and install ROS2 Jazzy + LeRobot environment
- [ ] Convert YOLO `.pt` to TensorRT `.engine`
- [ ] Install `onnxruntime-gpu` Jetson wheel for AlphaZero inference
- [ ] Replace macOS TTS with `espeak-ng` / `pyttsx3` (Linux equivalent)
- [ ] Deploy full stack on Jetson; profile with `tegrastats`
- [ ] Decide whether to deploy the learned policy on-Jetson (SmolVLA fits) or keep remote inference (π₀ / π₀.₅ require it) — see [`VLA_FINETUNING_PLAN.md`](./VLA_FINETUNING_PLAN.md)

#### 5.3 — Optional depth camera integration

The SO-101 Advanced Kit ships with two RGB cameras (wrist + scene) — depth is not strictly required. If a depth sensor is added later (e.g. RealSense D405) for richer perception:

- [ ] Sample board-plane depth at lock time
- [ ] Reject piece detections outside `board_near_mm` / `board_far_mm` range
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
| Phase 3B: VLA-ready simulation | ✅ ALGORITHM-COMPLETE | Geometry/IK/MoveIt/launch validated; sim grasp contact physics parked, deferred to real arm |
| Phase 3C: SO-101 hardware setup | 🔄 Arm ordered 2026-04-30 (ETA early June 2026); pre-arm prep in flight | Calibrate, collect demos, train ACT once arm arrives |
| Phase 4: Full game integration | 🔲 | SO-101 plays complete games vs. human |
| Phase 5: ROS2 vision integration + Jetson deployment | 🔲 DEPRIORITIZED | Untethered operation; Jetson buy trigger lives here |
| Phase 6: Polish | 🔲 | NLP, voice, web dashboard |

**Current focus:** Phase 3B closed out as algorithm-complete. SO-101 Advanced Kit ordered from Hiwonder on 2026-04-30 (lead time ~25 business days from China; ETA early June 2026). Active work is **Phase 3C prep** — see [`SESSION_STATUS.md`](./SESSION_STATUS.md) for the prioritized TODO list of pre-arm work (LeRobot environment, fixture printing, `arm_node` design, dataset schema).

**Hardware path summary:** SO-101 leader+follower → LeRobot ACT policy → trained on real demos → plays Connect Four. The UR5e simulation validated the algorithm; the SO-101 is the real-hardware endpoint at a fraction of the cost.
