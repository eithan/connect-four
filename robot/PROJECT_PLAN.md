# Connect Four Robot Arm — Full Project Plan

## Project Overview

Build a robot arm that can play Connect Four against a human opponent by:

Viewing a standard store-bought Connect Four board with a depth camera

- Detecting the board state and determining whose turn it is

- Calling into your existing AlphaZero AI model to choose the best move

- Physically picking up a piece and dropping it into the correct column

- Waiting for the human to play, then repeating

Your stack: Apple Silicon Mac (development) → Jetson Orin Nano (deployment) → ROS2 + Python + ONNX

## Robot Arm Recommendations

### Budget Tier (~$200–$400): Hiwonder ArmPi Ultra (Raspberry Pi version)

- Price: ~$300–$400 depending on configuration

- Specs: 6-DOF, bus servos (25 KG torque), 3D depth camera included, ROS2 compatible

- Pros: Comes with a depth camera out of the box, extensive tutorials, Python/OpenCV support, ROS2 ready. Affordable entry point.

- Cons: Raspberry Pi based (not Jetson), so you'd need to either use the Pi or swap controllers. Lower torque than JetArm. Smaller community than Elephant Robotics.

- Best for: Getting started quickly at low cost if you're okay with Raspberry Pi as the brain (or using it as a remote-controlled arm from your Jetson).

### Mid Tier (~$500–$800): Elephant Robotics myCobot 280

- Price: ~$650–$800 depending on version (M5Stack, Pi, or Jetson Nano variants)

- Specs: 6-DOF, 250g payload, 280mm reach, ±0.5mm repeatability, ~850g weight

- Pros: Largest community of any hobby arm, excellent ROS1/ROS2 + MoveIt support, Python/C++/C# APIs, 90+ control interfaces, drag-and-teach, LEGO-compatible ends. Jetson Nano version available. Huge ecosystem of accessories (grippers, cameras, suction cups). Well-documented Gitbook.

- Cons: 250g payload is light (Connect Four pieces are ~5g each, so this is fine, but limits future projects). No depth camera included — you'd add your own (e.g., Intel RealSense or the one from your Orin Nano setup). The Jetson Nano version uses the older Nano, not Orin Nano — you'd pair the M5Stack version with your existing Orin Nano as the brain.

- Best for: Best overall balance of community support, documentation, and ROS2 integration. Strong recommendation if you want a well-supported path.

### High Tier (~$700–$1,100+): Hiwonder JetArm (Jetson Orin Nano version)

- Price: ~$700–$1,100+ depending on kit tier (Advanced vs Ultimate) and Orin Nano memory (4GB vs 8GB). Note: this price includes the Jetson Orin Nano board.

- Specs: 6-DOF, bus servos (35 KG torque), 3D depth camera (Gemini) included, 6-mic array, ROS1 + ROS2, Gazebo simulation support, MoveIt, inverse kinematics source code provided

- Pros: Natively designed for Jetson Orin Nano — this is the arm built for your exact controller. Includes a 3D depth camera with RGB+D fusion. Comes with working examples for color recognition, AprilTag detection, 3D spatial grasping, sorting, and tracking. Full inverse kinematics source code and DH model provided. Gazebo simulation included. Comprehensive wiki and video tutorials. All-metal construction.

- Cons: Higher price (though it includes the Orin Nano). Hiwonder's ecosystem is less community-driven than Elephant Robotics — you're more reliant on their tutorials. The software stack is opinionated (their custom ROS packages).

- Best for: If you want the most turnkey "Jetson Orin Nano + arm + depth camera + ROS2" experience with the least integration work, this is it. Since you already use an Orin Nano, this is the path of least resistance.

### Honorable Mention: Seeed Studio reBot Arm B601

- Price: Target sub-$1,000 build cost (open source, you source parts)

- Specs: 6-DOF + gripper, 650mm reach, 1.5kg payload

**Recommendation:** Go with the Hiwonder JetArm (Orin Nano 8GB version) — specifically the "Advanced Kit with Orin Nano 8GB". It's purpose-built for your controller, includes the depth camera you need, and comes with working 3D grasping examples that are directly relevant to picking up Connect Four pieces. The 8GB Orin Nano provides 40 TOPS of AI performance (double the 4GB's 20 TOPS), 1024 CUDA cores, and 32 Tensor cores — critical headroom for running vision + AI inference + ROS2 + MoveIt concurrently, and essential for future natural language processing capabilities. The Elephant Robotics myCobot 280 is the runner-up — better community but more integration work since you'd need to add your own camera and bridge it to your Orin Nano.

**End effector:** Parallel gripper (the JetArm's default servo-driven claw). While a suction cup would be simpler for Connect Four discs specifically, the gripper is far more general-purpose and will support future tasks like picking up blocks, bottles, and other objects. The gripper requires more careful calibration for the thin Connect Four discs (~33mm diameter, ~8mm thick) but handles a much wider range of objects for future projects.

## Phased Project Plan

### Phase 1: Vision Pipeline on Mac (No ROS2, No Arm) ✅ COMPLETE

Goal: Build and test the computer vision system that can look at a Connect Four board image and extract the full board state.
Timeline: 2–3 weeks
Hardware needed: Just your Mac

#### Deliverables ✅
- `board_detector.py` — OpenCV board state extraction
- `turn_tracker.py` — Turn detection and game state management
- `ai_player.py` — ONNX model inference wrapper (AlphaZero, 3-channel input)
- `test_pipeline.py` — End-to-end test with sample images (8/8 perfect accuracy)
- `test_images/` — Synthetic test images with ground truth
- `requirements.txt` — Python dependencies

### Phase 2: Live Camera Feed on Mac

Goal: Replace static images with a live camera feed and validate real-time board detection.
Timeline: 1–2 weeks
Hardware needed: Mac + USB webcam (or Mac's built-in camera) + physical Connect Four board

#### 2.1 — Live Camera Capture

- Use OpenCV VideoCapture to stream from a webcam

- Add camera calibration for your specific setup (distance, angle, lighting)

- Implement frame rate throttling (no need to process every frame — 1–2 FPS is plenty)

#### 2.2 — Robust Detection Tuning

- Tune HSV thresholds for your specific lighting conditions

- Add adaptive thresholding or histogram equalization for varying light

- Handle partial occlusion (human's hand over the board while placing a piece)

- Add confidence scoring to board detection — only act on high-confidence frames

- Implement a "stable state" detector: only accept a new board state after N consecutive frames agree

#### 2.3 — Game Loop

- Build the full game loop:

  - Display "Waiting for human's move..."

  - Detect when exactly one new piece appears → human has played

  - Run AI inference → determine best column

  - Display "AI wants to play column X" (no arm yet — just prints/displays)

  - Wait for human to place the AI's piece (cooperative mode for now)

  - Detect the AI's piece was placed → go back to step 1

  - Handle game-over detection (win/draw) and reset

#### Deliverables

- `camera_feed.py` — Live capture with board detection overlay
- Updated `board_detector.py` with tuning parameters
- `game_loop.py` — Full game state machine
- Calibration utilities and documentation

### Phase 3: ROS2 Setup + Simulation

Goal: Port the vision pipeline to ROS2 nodes and set up a simulated robot arm in Gazebo.
Timeline: 3–4 weeks
Hardware needed: Mac (Docker or RoboStack for ROS2) or Jetson Orin Nano

#### 3.1 — ROS2 Environment Setup

- On Mac: Install ROS2 Jazzy via RoboStack (conda-based, works on Apple Silicon) or use Docker with GUI forwarding. RoboStack is recommended for development — it gives you native ROS2 + Gazebo in a conda environment.

- On Jetson Orin Nano: Install ROS2 Humble or Jazzy natively (Ubuntu-based, straightforward). This will be the deployment target.

- Verify: `ros2 topic list`, `rviz2`, basic pub/sub working

#### 3.2 — ROS2 Node Architecture

Create these ROS2 nodes (Python):

| Node | Subscribes To | Publishes | Purpose |
|------|--------------|-----------|---------|
| `camera_node` | — | `/camera/image_raw`, `/camera/depth` | Captures from depth camera |
| `board_detector_node` | `/camera/image_raw` | `/connect_four/board_state` | Detects board state from image |
| `game_manager_node` | `/connect_four/board_state` | `/connect_four/ai_move`, `/connect_four/game_status` | Tracks turns, calls AI, manages game state |
| `arm_controller_node` | `/connect_four/ai_move` | `/arm/joint_commands` | Translates "play column X" into arm motions |

- Define custom messages: `BoardState.msg` (int8[42] board, int8 current_player), `AiMove.msg` (int8 column)

- Each node is independently testable

#### 3.3 — Gazebo Simulation

- Load or create a URDF/SDF model of your chosen robot arm

  - If using JetArm: Hiwonder provides Gazebo models

  - If using myCobot 280: Elephant Robotics has URDF + MoveIt configs on GitHub

- Add a simulated Connect Four board to the Gazebo world (simple colored cylinders in a grid)

- Add a simulated depth camera on or near the arm

- Verify you can move the arm in simulation via ROS2 commands

- Use MoveIt 2 for motion planning — set up the arm's MoveIt config (kinematics solver, planning scene, collision objects)

#### 3.4 — Simulated Game Loop

- Wire up all nodes in simulation

- Use the simulated camera to detect the (simulated) board

- Manually place pieces in Gazebo to simulate human moves

- Verify the AI responds with correct column choices

- Arm moves to the correct column in simulation (doesn't need to actually grasp yet — just move to position)

#### Deliverables

- ROS2 workspace (`connect_four_ws/`) with all nodes
- Custom message definitions
- Gazebo world file with arm + board
- MoveIt 2 configuration for the arm
- Launch files for simulation and real hardware

### Phase 4: Arm Motion Planning

Goal: Teach the arm the specific motions needed for Connect Four: pick up a piece from a supply, move to the correct column, and drop it in.
Timeline: 2–3 weeks
Hardware needed: Robot arm (physical or simulated)

#### 4.1 — Define Key Poses

The arm needs these named poses/waypoints:

- **Home:** Resting position, out of the camera's view of the board

- **Piece supply:** Position above the spare pieces pile (keep spare pieces in a tray next to the board)

- **Grasp:** Lower to grab a piece from the supply

- **Column 0–6 approach:** Position above each of the 7 columns

- **Drop:** Release the piece into the column slot

#### 4.2 — Motion Sequences

For each AI move (column N):

1. Home → Piece supply approach (move above tray)
2. Lower → Grasp piece (close gripper)
3. Lift → Clear height (avoid hitting the board)
4. Move to Column N approach position (above the correct slot)
5. Release piece (open gripper — piece drops into slot)
6. Return to Home

- Use MoveIt 2 for path planning with collision avoidance (the board frame is a collision object)

- Define the board's physical location relative to the arm's base frame via calibration

- Each column's drop position is computed from the board's known geometry (columns are evenly spaced)

#### 4.3 — Gripper Strategy

- Connect Four pieces are small discs (~33mm diameter, ~8mm thick)

- Using parallel gripper (JetArm's default servo-driven claw) for general-purpose versatility

- Grip approach: come from the side and pinch the disc by its edges, or from above if pieces are standing upright in a supply tray

- Calibration needed: grip width (just wider than 33mm to approach, close to ~30mm to grip), grip force (firm enough to hold, gentle enough not to launch the piece)

- For the piece supply tray: orient spare pieces standing upright in a row so the gripper can approach from the front and pinch each one

#### 4.4 — Calibration

- Camera-to-arm calibration: Determine the transform between the camera frame and the arm's base frame. Use a checkerboard or AprilTag-based calibration.

- Board localization: Detect the board's position in 3D space using the depth camera. Mark the board's corners or use the grid itself as reference.

- Column positions: Compute the 3D coordinates of each column's drop point from the board localization.

#### Deliverables

- Named pose configurations (YAML)
- Motion sequence scripts for pick-and-place
- Gripper control integration
- Camera-to-arm calibration procedure
- Board localization and column mapping

### Phase 5: Integration + Real Hardware

Goal: Run the full system on the Jetson Orin Nano with the physical arm and board.
Timeline: 2–3 weeks
Hardware needed: Robot arm, Jetson Orin Nano, depth camera, Connect Four board, piece supply tray

#### 5.1 — Hardware Assembly

- Mount the arm on a stable surface

- Position the Connect Four board within the arm's reach

- Set up the piece supply tray (a simple 3D-printed or cardboard tray holding spare pieces in a known position)

- Mount the depth camera (fixed mount recommended for Connect Four — consistent top-down or angled view)

#### 5.2 — Deploy to Jetson Orin Nano

- Clone the ROS2 workspace to the Orin Nano

- Install dependencies (onnxruntime, OpenCV, MoveIt 2)

- If using JetArm: use their provided ROS2 packages and servo drivers

- If using myCobot: install the pymycobot SDK + ROS2 wrapper

- Update launch files for real hardware (real camera topic, real arm driver)

#### 5.3 — Tuning on Real Hardware

- Re-calibrate camera-to-arm transform with the physical setup

- Re-tune HSV thresholds for real lighting conditions

- Adjust grasp/release positions for the real piece dimensions

- Test drop accuracy: can the arm consistently drop pieces into each column?

- Tune MoveIt speed/acceleration for smooth motion

#### 5.4 — Full Game Test

- Play a complete game against the robot

- Verify the full loop: human plays → camera detects → AI decides → arm picks up piece → arm drops piece → waits for human

- Handle edge cases: piece bounces out, piece supply runs out, board knocked out of position

- Add error recovery: if board state doesn't match expectations, pause and alert

#### Deliverables

- Fully deployed ROS2 system on Jetson Orin Nano
- Hardware calibration data
- Full game demonstration video
- Error handling and recovery procedures

### Phase 6: Polish + Advanced Features (Optional)

Goal: Make the system robust, user-friendly, and impressive.
Timeline: Ongoing

#### 6.1 — Natural Language Commands (Primary Enhancement)

Architecture:

- **Speech-to-text:** Whisper model running on Orin Nano with TensorRT acceleration. The JetArm's 6-mic array captures audio.

- **Intent parsing:** An LLM (local small model or API call) extracts structured commands from natural language.

- **Visual grounding:** The depth camera locates the referenced object and destination in 3D space.

- **Motion execution:** MoveIt plans and executes the pick-and-place based on the 3D coordinates.

#### 6.2 — Other Potential Enhancements

- Voice feedback: "Your turn!" / "I'm thinking..." / "I win!" using text-to-speech

- LED indicators: Light up the column the AI is about to play

- Automatic board reset: Arm pulls the board's release slider to dump all pieces

- Difficulty levels: Vary the MCTS simulation count to make the AI easier or harder

- Web dashboard: Stream the camera feed + game state to a web page

- Multiple game support: Teach the same arm to play tic-tac-toe, checkers, etc.

## Summary Timeline

| Phase | Duration | What You Need |
|-------|----------|---------------|
| Phase 1: Vision on Mac (static images) ✅ | 2–3 weeks | Mac only |
| Phase 2: Live camera on Mac | 1–2 weeks | Mac + webcam + physical board |
| Phase 3: ROS2 + Simulation | 3–4 weeks | Mac (RoboStack/Docker) + Jetson |
| Phase 4: Arm motion planning | 2–3 weeks | Arm (simulated or physical) |
| Phase 5: Real hardware integration | 2–3 weeks | Full hardware setup |
| Phase 6: Polish | Ongoing | Everything |

Total estimated time: 10–15 weeks (part-time/hobby pace)

**When to order the arm:** During Phase 2. By then you'll have validated your vision pipeline works and you'll be ready to start simulation. Shipping from Hiwonder/Elephant Robotics typically takes 1–3 weeks.
