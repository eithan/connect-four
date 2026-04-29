# Connect Four Robot Arm — Alternative Plan (LeRobot SO-101)

**Status:** Alternative path. Use this if the JetArm doesn't work out, or if the LeRobot/VLA learning goal becomes the priority. The primary plan is in [`ROBOT_PLAN.md`](./ROBOT_PLAN.md). Last revised 2026-04-28.

This document mirrors the structure of `ROBOT_PLAN.md` but assumes the **HuggingFace LeRobot SO-ARM101** instead of the Hiwonder JetArm. Most of the architecture above the arm is identical (existing AlphaZero + vision + game state on Jetson, ROS2 orchestration, single `PlacePiece.action` contract). The differences live below the action server: a different SDK, more fabrication, and a much richer downstream path to imitation learning and VLA fine-tuning.

---

## 1. Goals (same as primary plan, with one shift)

1. Build a robot that plays a full game of Connect Four against a human on a stock vertical Hasbro board, end to end.
2. Reuse the existing AlphaZero ONNX model and vision pipeline without rewriting them.
3. Learn the **HuggingFace LeRobot stack** end-to-end: teleoperation, demo recording, ACT/Diffusion/SmolVLA fine-tuning, deployment.
4. Use ROS2 as the orchestration layer above LeRobot (so the ROS2 learning goal stays in scope).
5. Keep imitation-learning and VLA fine-tuning as **first-class v2 milestones**, not afterthoughts.

The defining shift versus the JetArm plan: **VLA / imitation learning is no longer optional polish — it's the reason this path is chosen.** If you don't intend to do VLA work, the JetArm plan is the better choice on every other axis.

---

## 2. Decisions Locked (under this plan)

| Decision | Choice | Rationale |
|---|---|---|
| Robot arm | **LeRobot SO-ARM101 (leader + follower kit, fully assembled)** | Canonical LeRobot hardware. Native HuggingFace ecosystem support; SmolVLA / π0 / OpenVLA fine-tuning recipes target this exact arm. Leader/follower teleop pattern is built in. |
| Compute | **Jetson Orin Nano 8GB Super Dev Kit** | Same as primary plan. Runs onnxruntime-gpu inference plus modest VLA fine-tuning. |
| Storage | **256GB NVMe SSD** | Same as primary plan. Tight if heavy demo recording happens — revisit if you commit to VLA training on-device. |
| Cameras | **Logitech C920 overhead + USB endoscope wrist cam** | Two RGB views. No depth sensor in the base plan; add one later if needed. |
| Game piece delivery | **Top-of-column drop** | Same as primary plan. |
| Software approach v1 | **Hybrid: scripted arm motion (LeRobot SDK), learned vision/AI** | Get a working game first. |
| Software approach v2 | **Behavior-cloned or VLA-driven arm motion** | Replace scripted motion with a learned policy fine-tuned on teleop demos. This is the SO-101 path's reason for being. |
| Orchestration | **ROS2 Humble on Ubuntu 22.04** wrapping LeRobot Python | Same orchestration shape as primary plan. ROS2 actions sit above LeRobot SDK calls. |
| Workspace | **3D-printed riser (~12 cm) + custom TPU gripper fingertips + structured piece tray**, all printed on the Flashforge Adventurer 3 | SO-101's reach is borderline on a vertical Hasbro board without a riser; stock gripper jaws are not optimal for 40 mm discs. |

### Decisions explicitly not taken

- **Not buying:** xArm Lite 6 / Niryo Ned2. Better arms but blow the budget.
- **Not buying:** JetArm. Considered (and the primary plan); switched away from it because LeRobot/VLA path is the goal here.
- **Not skipping:** The leader arm. The leader+follower kit costs only ~$50–80 more than follower-only and earns its keep at the moment teleop demo recording starts.

---

## 3. Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                 Jetson Orin Nano 8GB (Ubuntu 22.04 + ROS Humble)   │
│                                                                    │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────────────┐   │
│  │ vision_node  │──▶│  game_node   │──▶│   policy_node        │   │
│  │ (existing)   │   │ (port from   │   │   (AlphaZero ONNX,   │   │
│  │ overhead +   │   │  web/src/    │   │    existing model)   │   │
│  │ wrist RGB    │   │  game/       │   │                      │   │
│  │              │   │  connectFour │   │                      │   │
│  └──────┬───────┘   │  .js)        │   └──────────┬───────────┘   │
│         │           └──────────────┘              │               │
│         │                                         ▼               │
│         │           ┌─────────────────────────────────────┐       │
│         │           │       orchestrator_node             │       │
│         └──────────▶│  (game state machine, turn          │       │
│                     │   detection, move dispatch)         │       │
│                     └────────────────────┬────────────────┘       │
│                                          │                        │
│                                          │ /place_piece           │
│                                          │   (column N)           │
│                                          ▼                        │
│                     ┌─────────────────────────────────────┐       │
│                     │           arm_node                  │       │
│                     │  ROS2 action server                 │       │
│                     │  PlacePiece.action                  │       │
│                     │  ─ wraps LeRobot Python SDK         │       │
│                     │  ─ v1: scripted keyframe replay     │       │
│                     │  ─ v2: ACT / Diffusion / SmolVLA    │       │
│                     └────────────────┬────────────────────┘       │
└──────────────────────────────────────┼────────────────────────────┘
                                       │ USB (Feetech STS3215 bus)
                                       ▼
                       ┌─────────────────────────────────────┐
                       │  LeRobot SO-ARM101 hardware         │
                       │   ┌──────────────┐  ┌────────────┐  │
                       │   │ FOLLOWER ARM │  │ LEADER ARM │  │
                       │   │  6 servos    │  │  6 servos  │  │
                       │   │  (gripper +  │  │  (used for │  │
                       │   │   5-axis)    │  │   teleop)  │  │
                       │   └──────────────┘  └────────────┘  │
                       │  Mounted on 3D-printed riser        │
                       └─────────────────────────────────────┘
```

The action contract is the same as in the primary plan — `PlacePiece.action` with `column`, `success`, `error_code`, `phase`, `progress`. This means the AI/vision/orchestration layer is identical to the JetArm path; only `arm_node`'s implementation changes underneath.

### v2 architecture: learned arm policy

When transitioning to a learned policy, only `arm_node` changes:

```
arm_node (v2)
  ├── On goal received: read current camera + joint state
  ├── Run policy (ACT / Diffusion / SmolVLA) → produce action chunk
  ├── Stream actions to follower at 30–50 Hz
  └── Monitor success criteria (grasp confirmation, drop confirmation)
```

The policy is trained offline on demos recorded via the leader arm (Phase C).

---

## 4. Hardware

### Shopping list

| Item | Vendor (typical) | Cost |
|---|---|---|
| SO-ARM101 leader + follower kit, fully assembled | Seeed Studio / WowRobo / Hiwonder | ~$470 |
| Jetson Orin Nano 8GB Super Dev Kit | NVIDIA / Amazon / Arrow | ~$249 |
| 256GB NVMe SSD | Amazon / Newegg | ~$75 |
| Logitech C920 (overhead) | Amazon | ~$70 |
| USB endoscope camera or small CSI cam (wrist) | Amazon | ~$30 |
| TPU 95A filament + PLA+ for riser | Amazon | ~$50 |
| M3 brass heat-set inserts + M3 hardware | Amazon | ~$20 |
| Misc cables, USB-C hub, power adapters | — | ~$50 |
| Tax + shipping | — | ~$80 |
| **Total** | | **~$1,094** |

Comfortably under the $1,500 ceiling. Roughly $40 cheaper than the JetArm path; the savings come from the cheaper arm minus the cost of cameras and printing materials that the JetArm bundles for free.

### What the SO-ARM101 kit ships with (verify before ordering)

The "leader + follower kit, assembled" variant from Seeed/WowRobo/Hiwonder typically includes:

- 2× fully assembled 6-servo arms (leader + follower)
- Feetech STS3215 servos throughout (with magnetic encoders, ~0.088° resolution)
- Power supply (12V)
- USB-to-serial adapter for the Feetech bus
- Mounting hardware
- Stock parallel-jaw gripper

Does NOT include: cameras, riser, custom fingertips, piece tray. All on us to source or print.

### Workspace fabrication (all 3D-printed on the Flashforge Adventurer 3, build vol. 150 × 150 × 150 mm)

| Part | Material | Notes |
|---|---|---|
| Arm riser (~120 × 120 × 120 mm hollow box) | PETG or PLA+ | 30–40% gyroid infill, 4 mm walls, internal cross ribs at the top platform, M3 brass heat-set inserts for arm mounting. Optional sand fill in bottom cavity for damping. |
| Custom gripper fingertips | TPU 95A | Concave cradle sized to ~40 mm OD disc edge. Iterate based on grasp tests. |
| Piece tray | PLA+ | 7+ structured pockets so pickup pose is fixed and known. Slight chamfer on each pocket so the gripper self-centers. |
| Optional: board pedestal | PLA+ | Lifts the Hasbro board ~5 cm if reach turns out to be tight in practice. |

---

## 5. Open Questions & Risks (SO-101-specific)

1. **Vertical board reach** — SO-101 reach (~340 mm horizontal, ~420 mm vertical at full extension) is borderline for a 24 cm vertical Hasbro board. The riser is the standard mitigation; verify by measurement once parts arrive.
2. **Wrist backlash on direction reversal** — STS3215 servos with plastic gears have ~1–2° wrist play. Hide it by always approaching the column from one consistent direction.
3. **Repeatability under load** — Empty arm: ±1–2 mm at the tip. Holding a disc: ±2–5 mm typical. C4 columns are 30 mm wide with 37 mm pitch → ~13 mm margin per side. Workable but not lavish.
4. **ROS2 driver maturity** — Community-maintained (e.g. `JafarAbdi/lerobot_ros2`, various `ros2_so_arm100` forks). Less polished than Hiwonder's official ROS2 stack. We'll bridge from LeRobot Python rather than rely on a full ROS2-native driver.
5. **Smoothness gap vs. the JetArm** — Visible. SO-101 is plastic-bushed, hobbyist-grade. Mitigate via `ruckig` jerk-limited trajectories streamed at 100 Hz, PID tuning, and consistent approach directions. It will not look like a UR3e even after tuning.
6. **Teleop leader feel** — The leader arm is itself a Feetech-servo arm with backdrive enabled. Feel is good in published demos but varies with calibration; expect to tune.

---

## 6. Plan

Three phases, parallel to the primary plan.

### Phase A — Jetson-first work + heavy fabrication (~1.5–2 weeks, while arm ships)

This phase has more pre-arm work than the JetArm plan because there's significant 3D printing and CAD to do. Use the wait time well.

#### A.1 Jetson bring-up
- [ ] Mount 256GB NVMe SSD in Jetson Orin Nano 8GB Super Dev Kit
- [ ] Flash JetPack 6.x (Ubuntu 22.04 base)
- [ ] First boot, network, SSH
- [ ] Install ROS2 Humble desktop and dev tools (colcon, rosdep, vcstool)
- [ ] Install Python deps: `onnxruntime-gpu`, `numpy`, `opencv-python`, `pyserial`, `ruckig`
- [ ] Verify CUDA + TensorRT install

#### A.2 LeRobot dev environment
- [ ] Create a venv or conda env for LeRobot work
- [ ] `pip install lerobot` from HuggingFace (latest v0.5.0+)
- [ ] Install PyTorch with CUDA support matching the JetPack TensorRT version
- [ ] Smoke-test by loading a published SO-101 dataset from the HuggingFace Hub
- [ ] Skim the LeRobot tutorial notebooks (teleop, recording, training, inference)

#### A.3 Existing code on Jetson
- [ ] Clone `eithan/connect-four` repo onto Jetson
- [ ] Run `alphazero-network-model.onnx` via `onnxruntime-gpu` with CUDA EP; benchmark inference latency
- [ ] If latency matters: try TensorRT EP via onnxruntime, then native TensorRT engine
- [ ] Port `web/src/game/connectFour.js` rules engine to Python (`game_node` core logic)
- [ ] Confirm existing Connect Four vision + game-detection code runs

#### A.4 ROS2 workspace skeleton
- [ ] Create `ros2_ws/src/connect_four_arm/` package (ament_python)
- [ ] Define `PlacePiece.action` interface in `ros2_ws/src/connect_four_msgs/`
- [ ] Implement `arm_node` as a stub: accept goals, publish feedback every 200 ms, sleep through fake motion phases, return success
- [ ] Implement `orchestrator_node`: subscribes to `/board_state`, runs game loop, calls `arm_node` action
- [ ] Implement `policy_node`: wraps ONNX inference; publishes chosen column on game state change
- [ ] Wire stub `vision_node`: publishes a synthetic `/board_state` from a fake game progression for testing
- [ ] Add `launch/connect_four.launch.py` that brings up all five nodes
- [ ] Add a basic README under `ros2_ws/`

#### A.5 Fabrication (the big pre-arm task on this path)
- [ ] CAD the arm riser (~120 × 120 × 120 mm hollow box, ribbed top platform with M3 inserts)
- [ ] CAD a structured piece tray with 7+ pockets sized to ~40 mm OD discs
- [ ] CAD initial TPU fingertip prototype with a concave cradle for the disc edge — reference existing LeRobot community CAD
- [ ] Print riser in PETG/PLA+ on Flashforge Adventurer 3
- [ ] Print piece tray in PLA+
- [ ] Print TPU fingertips in TPU 95A (slow speed, 230–240 °C)
- [ ] Install brass heat-set inserts in riser top platform
- [ ] Have all parts ready to mount the arm on day 1 of Phase B

#### A.6 Stub end-to-end test
- [ ] With all nodes running and a synthetic board state, verify orchestrator → policy → arm flow completes a fake game
- [ ] Replay scenarios via `ros2 bag` to validate failure paths

#### A.7 Documentation prep
- [ ] Read LeRobot docs front to back, focusing on: teleop, dataset recording, dataset format, ACT/Diffusion training, SmolVLA fine-tuning
- [ ] Skim community ROS2 SO-100/101 driver repos to choose a starting point for the Phase B bridge

**Phase A success criterion:** Synthetic game runs end-to-end through ROS2 nodes on Jetson; AlphaZero ONNX inference confirmed working on GPU; LeRobot dev environment ready; all 3D-printed workspace parts (riser, fingertips, piece tray) are physically in hand.

---

### Phase B — Hardware integration (~1.5 weeks, once arm arrives)

Replace the stub arm node with real SO-101 motion. Introduce real cameras + vision. Play a real game.

#### B.1 Unboxing + verification
- [ ] Inspect both arms (leader + follower) for shipping damage
- [ ] Connect via USB; run LeRobot's calibration UI for both arms — this is the canonical first step
- [ ] Run LeRobot's teleoperation demo (leader → follower) to verify the full chain: motors, encoders, USB-to-serial adapter
- [ ] Resolve any servo ID conflicts or calibration drift before proceeding

#### B.2 Workspace install
- [ ] Bolt arm to the printed riser using the heat-set inserts
- [ ] Position riser, Connect Four board, and piece tray on a flat surface; clamp/screw down so positions don't drift
- [ ] Mount the Logitech C920 overhead with a clear view of the board face
- [ ] Mount the wrist camera on the gripper fixture
- [ ] Document all positions in `ros2_ws/src/connect_four_arm/config/geometry.yaml`

#### B.3 Custom gripper fingertips
- [ ] Swap stock fingertips for the TPU prints from Phase A
- [ ] Tune grip force in LeRobot calibration UI
- [ ] Pick-and-place test: 50 trials grasping a disc from the tray, lifting, and replacing — measure success rate and any slipping
- [ ] Iterate on the TPU print if reliability is below ~95% (typically 1–2 print cycles)

#### B.4 Scripted pick-and-place via LeRobot SDK
- [ ] Use the leader arm to teleoperate the follower into 8 keyframe poses:
  - `home`
  - `pickup_tray`
  - `drop_col_0` … `drop_col_6`
- [ ] Save keyframes (joint positions) to `geometry.yaml`
- [ ] Write a standalone Python script using the LeRobot follower API: loop `home → pickup_tray → drop_col_N → home` for each N
- [ ] Layer in `ruckig` jerk-limited trajectories streamed to the follower at 100 Hz — this is the smoothness pass; do not skip
- [ ] Bias all column approaches to come from the same direction so wrist backlash never reverses mid-motion
- [ ] Tune STS3215 PID + `goal_acc` per joint
- [ ] **Success criterion:** ≥95% successful drops in correct column over 50 trials with motion that looks deliberate, not jittery

#### B.5 Real `arm_node`
- [ ] Replace stub `arm_node` motion with calls to the Phase B.4 motion library
- [ ] Action interface (`PlacePiece.action`) stays unchanged
- [ ] Add servo health/error reporting via Feetech telemetry (position error, voltage, temperature)
- [ ] Add timeouts and joint-limit pre-checks
- [ ] Add `/estop` topic
- [ ] Implement a "failed grasp" detector using STS3215 torque/current proxy on close-gripper

#### B.6 Real `vision_node`
- [ ] Calibrate Logitech C920 intrinsics + extrinsics to the locked board position
- [ ] Port existing Connect Four vision + game-state code to publish `/board_state`
- [ ] Add wrist camera as a secondary view — optional for v1, used in Phase C for VLA work

#### B.7 End-to-end games
- [ ] Full game loop with a human opponent
- [ ] Iterate on failure modes
- [ ] **Success criterion:** Robot completes ≥3 full games against a human without intervention

**Phase B success criterion:** Functional Connect Four robot, demo-able. Same bar as the JetArm plan; more sweat to get there.

---

### Phase C — LeRobot stack payoff (this is why you chose this path)

Phase C is where the SO-101 path differentiates from the JetArm plan. Pick what's useful.

#### C.1 Dataset recording (foundation for everything below)
- [ ] Use LeRobot's `record` script to capture teleoperated Connect Four games
- [ ] Record 50–100 episodes: full pick-and-place sequences across all 7 columns and varying piece-tray states
- [ ] Each episode captures: leader joint commands, follower joint states, both camera streams, timestamps
- [ ] Push dataset to HuggingFace Hub (private repo) for portability and visualization

#### C.2 Behavior cloning baseline (ACT or Diffusion Policy)
- [ ] Fine-tune **ACT** on the recorded dataset (small, fast, well-documented in LeRobot)
- [ ] Train on a desktop GPU; deploy weights back to Jetson for inference
- [ ] Build an `arm_node_v2` that uses the trained policy instead of scripted keyframes
- [ ] Compare to scripted baseline: success rate, motion smoothness, generalization to small piece-tray variations

#### C.3 SmolVLA fine-tuning (the marquee experiment)
- [ ] Take the same dataset + add language annotations: "place a red piece in column 3"
- [ ] Fine-tune **SmolVLA** (450M params, designed for SO-100/101) on the annotated dataset
- [ ] Deploy on Jetson; benchmark inference latency vs. ACT
- [ ] Compare task success and generalization
- [ ] Optional: explore zero-shot transfer to slight variations (different piece colors, board positions)

#### C.4 Other polish + stretch
- [ ] **MoveIt2 motion planning** as an alternative to scripted/learned arm motion
- [ ] **URDF + Gazebo simulation** of the SO-101 (community URDFs available)
- [ ] **Gamepad teleop** as a fallback when leader arm isn't available
- [ ] **3D-printed gripper variants** (e.g. soft/compliant fingers, finger-cam holders)
- [ ] **Switch to a depth camera** (RealSense D405 or similar) — LeRobot supports it
- [ ] **Publish dataset + writeup** on HuggingFace Hub as a contribution to the LeRobot community

---

## 7. Existing Code Reuse

Identical to the primary plan. From [`eithan/connect-four`](https://github.com/eithan/connect-four):

| Existing | Reuse strategy |
|---|---|
| `ai/src/connect_four_ai/models/alphazero-network-model.onnx` | Loaded by `policy_node` via `onnxruntime-gpu` |
| `ai/` — AlphaZero training pipeline | Offline only; not deployed at runtime |
| `web/src/game/connectFour.js` rules engine | Port to Python in `game_node` |
| Existing Connect Four vision / detection code | Reused as the basis for `vision_node` |
| `web/` React frontend | Untouched |

The robot-side code is purely additive.

---

## 8. Repository Additions

Same proposed structure as the primary plan, with the addition of LeRobot-related directories:

```
connect-four/
├── ai/                          # existing AlphaZero training (unchanged)
├── web/                         # existing React frontend (unchanged)
├── docs/
│   ├── ROBOT_PLAN.md           # primary (JetArm) plan
│   └── ROBOT_PLAN_ALT.md       # ← this file
├── robot/                       # NEW: Python helpers
│   ├── calibrate.py
│   ├── teleop_capture.py
│   ├── motion.py                # ruckig-based trajectory generation
│   ├── lerobot_glue/            # adapters between LeRobot and our motion lib
│   └── tests/
├── lerobot_data/                # NEW: recorded teleop datasets (gitignored)
│   └── README.md                # describes dataset, points at HF Hub
├── policies/                    # NEW: trained policy weights (gitignored or LFS)
└── ros2_ws/                     # NEW: ROS2 workspace
    └── src/
        ├── connect_four_msgs/
        ├── connect_four_arm/
        └── connect_four_bringup/
```

---

## 9. Comparison with Primary Plan (JetArm)

A quick reference for deciding between paths.

| Dimension | Primary (JetArm) | Alternative (SO-101) |
|---|---|---|
| Total cost | ~$1,134 | ~$1,094 |
| Pre-arm fabrication | None | Riser + tray + fingertips (~3D-print weekend) |
| Out-of-box motion smoothness | Better (metal frame, industrial bearings) | Worse, requires explicit tuning |
| Reach for vertical Hasbro board | Comfortable (53 cm built-in height) | Borderline; needs 12 cm riser |
| ROS2 native support | Yes (Hiwonder official) | Community-maintained bridge |
| Cameras included | Orbbec Gemini Plus depth + 6-mic + touchscreen | None — buy separately |
| Leader/follower teleop | Not natively supported | Built-in (canonical use case) |
| Imitation learning ecosystem | Hiwonder's own (smaller community) | HuggingFace LeRobot (large, active) |
| VLA fine-tuning path | Off the paved road | First-class (SmolVLA / π0 / OpenVLA) |
| Demo polish ceiling | Higher | Lower without significant tuning effort |
| Right choice if your goal is… | Working demo + general ROS2 platform | Learning LeRobot + VLA experimentation |

If after Phase A on the JetArm path you find yourself yearning for the LeRobot ecosystem, you can buy a follower-only SO-101 (~$300) later as a second platform without abandoning anything.

---

## 10. References & Sources

- [`eithan/connect-four`](https://github.com/eithan/connect-four) — the existing repo this plan extends
- [HuggingFace LeRobot — GitHub](https://github.com/huggingface/lerobot)
- [HuggingFace LeRobot — Docs](https://huggingface.co/docs/lerobot)
- [LeRobot v0.5.0 release notes](https://huggingface.co/blog/lerobot-release-v050)
- [SmolVLA blog post](https://huggingface.co/blog/smolvla)
- [SmolVLA model card](https://huggingface.co/lerobot/smolvla_base)
- [SO-ARM100 / SO-ARM101 hardware repo (TheRobotStudio)](https://github.com/TheRobotStudio/SO-ARM100)
- [Hiwonder LeRobot SO-ARM101 product page](https://www.hiwonder.com/products/lerobot-so-101)
- [Seeed Studio SO-ARM100/101 listing](https://www.seeedstudio.com/SO-ARM100-low-cost-AI-arm-kit-pre-assembled-p-6343.html)
- [`ruckig` — jerk-limited trajectory generation](https://github.com/pantor/ruckig)
- [MoveIt2 docs](https://moveit.picknik.ai/main/index.html)
- [NVIDIA Jetson Orin Nano Super Developer Kit](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/nano-super-developer-kit/)
- [ROS 2 Humble docs](https://docs.ros.org/en/humble/)

---

## 11. Changelog

- **2026-04-28** — Initial version. Parallel alternative to `ROBOT_PLAN.md`. Three-phase plan with Phase A heavy on fabrication and LeRobot environment setup; Phase C centered on imitation learning + SmolVLA fine-tuning as the path's payoff.
