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

### Phase 3B: VLA-Ready Simulation 🔲 NEXT

Goal: Upgrade the Gazebo scene to be physically accurate and capable of collecting VLA training demonstrations. Success = arm can execute a scripted pick-place-release sequence in sim with realistic physics.

#### 3B.1 — Realistic Gazebo Scene
- [ ] Add a table/surface at the correct height in the Gazebo world SDF
- [ ] Create Connect Four board collision mesh (292mm × 254mm × 64mm box with 7 column slots — simplified as a solid box with a slot top surface)
- [ ] Create Connect Four piece SDF models: red and yellow cylinders (31mm diameter × 7mm thick), correct mass/inertia, appropriate friction coefficients
- [ ] Spawn 7 pieces of each color in a supply tray next to the board at known positions within the arm's workspace
- [ ] Verify pieces fall and stack realistically in Gazebo physics

#### 3B.2 — Robotiq 2F-85 Gripper Integration
- [ ] Add `robotiq_description` package to the workspace (ROS2 Jazzy compatible fork)
- [ ] Attach Robotiq 2F-85 URDF to UR5e `tool0` link
- [ ] Configure gripper MoveIt group and controllers in `connect_four_moveit_controllers.yaml`
- [ ] Verify gripper open/close in Gazebo via ROS2 action interface
- [ ] Test that closed fingers can contact and hold a piece cylinder (physics contact working)

#### 3B.3 — Full Pick-Place-Release Sequence
- [ ] Define pick pose for each supply tray position (above piece → descend → grasp → lift)
- [ ] Precompute IK for supply tray positions (same approach as column IK)
- [ ] Extend `column_mover.py` (or new node) to accept "pick piece color X, drop in column N" commands
- [ ] Full motion sequence: home → above_piece → descend → grasp → lift → above_column → descend → release → home
- [ ] Test all 7 columns with both red and yellow pieces

#### 3B.4 — Scene Camera Setup
- [ ] Mount a wrist camera on the UR5e (Intel RealSense D435 or similar SDF model)
- [ ] Mount an overhead/scene camera fixed in the Gazebo world
- [ ] Publish camera topics from Gazebo plugins
- [ ] Verify images show gripper, pieces, and board clearly — these will be VLA observations

#### Deliverables
- Updated Gazebo world SDF with table, board, pieces, cameras
- Connect Four piece SDF models (red + yellow)
- UR5e + Robotiq 2F-85 combined URDF
- Extended `column_mover.py` with full pick-place sequence
- RViz2 config updated to show gripper and piece markers

---

### Phase 3C: VLA Training & Deployment 🔲

Goal: Train a Vision-Language-Action model that can interpret a natural language command ("Pick up a red piece and drop it in column 1") and execute it in simulation. Success = command works reliably in sim across varied piece positions.

#### 3C.1 — Demonstration Collection
- [ ] Build a scripted demonstration collector: runs the pick-place sequence with small random offsets to piece positions (±5–10mm), records joint trajectories + wrist camera images + language command
- [ ] Collect ~100–200 demonstrations per piece color (200–400 total)
- [ ] Store demonstrations in LeRobot or RLDS format (standard for VLA training)
- [ ] Validate demonstration quality: replay a subset and verify they look correct

#### 3C.2 — VLA Model Selection & Fine-Tuning

Candidate models (in order of recommendation):
- **OpenVLA** (Stanford) — open-source, strong language grounding, fine-tunable on custom demos
- **ACT (Action Chunking Transformer)** — simpler, fast inference, great for repetitive tasks
- **Diffusion Policy** — best for contact-rich tasks, higher compute

Recommended path: Start with ACT (fastest to get working), then upgrade to OpenVLA for language grounding.

- [ ] Set up training environment (CUDA, PyTorch, model-specific dependencies)
- [ ] Format demonstration data for chosen model
- [ ] Fine-tune model on collected demonstrations
- [ ] Evaluate: does the model successfully pick the correct color piece and drop in the correct column?

#### 3C.3 — Sim Deployment
- [ ] Build a ROS2 node that takes a natural language command, runs VLA inference, and publishes joint trajectory commands
- [ ] Test in Gazebo: "Pick up a red piece and drop it in column 3"
- [ ] Evaluate success rate across all 7 columns and both colors

#### Deliverables
- Demonstration dataset (joint trajectories + images + language commands)
- Fine-tuned VLA model checkpoint
- ROS2 VLA inference node
- Evaluation report (success rate per column/color)

---

### Phase 4: Real Hardware Integration 🔲

Goal: Transfer the working simulation to a real robot arm. The VLA model trained in sim should transfer with minimal additional calibration.

#### 4.1 — Hardware Assembly
- [ ] Mount chosen arm on a stable surface
- [ ] Position Connect Four board within arm's reach (matching sim geometry: board center at ~650mm from base, table height matching sim)
- [ ] Set up piece supply tray at simulated position
- [ ] Mount wrist camera and overhead camera matching sim placement

#### 4.2 — ROS2 Driver Setup
- [ ] Install arm-specific ROS2 driver (UR5e driver or arm vendor package)
- [ ] Verify arm responds to MoveIt2 commands
- [ ] Install Robotiq 2F-85 ROS2 driver (or equivalent real gripper driver)
- [ ] Verify gripper open/close on real hardware

#### 4.3 — Sim-to-Real Calibration
- [ ] Camera extrinsic calibration (wrist camera to tool0 frame, overhead camera to base frame)
- [ ] Verify piece positions in real tray match sim coordinates (adjust `COLUMN_POSES` and tray pose if needed)
- [ ] Tune gripper closing force and width for real Connect Four pieces
- [ ] Test drop accuracy: does piece land in the correct column slot?

#### 4.4 — VLA Sim-to-Real Transfer
- [ ] Run VLA model in real environment; evaluate success rate
- [ ] Collect ~20–30 real-world demonstrations if sim-to-real gap is too large
- [ ] Fine-tune VLA on mixed sim + real data if needed

#### 4.5 — Full Game Test
- [ ] Wire up `game_loop.py --ros` with real hardware
- [ ] Play complete games end-to-end: camera detects board → AI decides column → VLA picks piece + drops
- [ ] Handle edge cases: piece supply runs out, dropped piece bounces, board shifts

#### Deliverables
- Real hardware calibration data (camera transforms, tray positions)
- Sim-to-real tuning notes
- Full game demonstration video

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
| Phase 3B: VLA-ready simulation | 🔲 NEXT | Pieces + gripper + pick-place in sim |
| Phase 3C: VLA training | 🔲 | OpenVLA/ACT fine-tuning + sim deployment |
| Phase 4: Real hardware | 🔲 | Physical arm + sim-to-real transfer |
| Phase 5: Vision integration | 🔲 DEPRIORITIZED | ROS2 board detection, Jetson, Orbbec |
| Phase 6: Polish | 🔲 | NLP, voice, web dashboard |

**Current focus: Phase 3B** — get realistic pieces, board, and Robotiq 2F-85 gripper into Gazebo so the arm can physically interact with the world.

**When to order the arm:** Any time during Phase 3B/3C. By the time the VLA model is trained in sim you'll be ready to transfer to hardware. The UR5e + Robotiq 2F-85 used in simulation is industry-standard hardware if budget allows; for a cheaper option the Elephant Robotics myCobot 280 + compatible parallel gripper is the best community-supported path.
