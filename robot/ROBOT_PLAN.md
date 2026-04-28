# Connect Four Robot Arm — Plan

**Status:** Source of truth. Update this file as decisions change. Last revised 2026-04-28.

A robotic-arm extension to the existing [`eithan/connect-four`](https://github.com/eithan/connect-four) project. The arm physically picks up Connect Four pieces and drops them into a stock Hasbro board, driven by the existing AlphaZero AI and vision pipeline running locally on a Jetson Orin Nano.

---

## 1. Goals

1. Build a robot that plays a full game of Connect Four against a human on an unmodified vertical Hasbro board, end to end.
2. Reuse the existing AlphaZero ONNX model and vision/board-state pipeline from this repo without rewriting them.
3. Learn the ROS2 stack (Humble, MoveIt2, URDF, Gazebo) properly along the way, with a system that can be reused for future, more complex robotic tasks.
4. Keep the path to imitation-learning experiments open, but don't pay any cost for it in v1.

Non-goals for v1: VLA fine-tuning, leader/follower teleoperation, multi-arm setups, novel research contributions.

---

## 2. Decisions Locked

| Decision | Choice | Rationale |
|---|---|---|
| Robot arm | **Hiwonder JetArm Ultimate Kit (without Jetson)** | Metal aluminum frame, ~53 cm tall integrated base (no riser needed for vertical board), Orbbec Gemini Plus depth camera included, native ROS2 Humble support, official URDF + MoveIt2 examples, comprehensive tutorials. |
| Compute | **Jetson Orin Nano 8GB Super Dev Kit** | ~67 TOPS at $249, comfortably runs onnxruntime-gpu inference for the existing AlphaZero net plus vision pipeline; headroom for future learned policies. |
| Storage | **256GB NVMe SSD** | The Hiwonder kit ships with 128GB SSD bundled with their bundled Jetson option, but since we're buying the Jetson separately, we add our own. |
| Camera | **Orbbec Gemini Plus** (included with arm) | RGB+D enables better board-state recognition than 2D alone, banked for future tasks. |
| Game piece delivery | **Top-of-column drop** | Pick from a structured tray, position above target column, open gripper, gravity drops the piece. |
| Software approach | **Hybrid: scripted arm motion, learned vision/AI** | Vision and AlphaZero already exist and work; the arm side is geometrically simple (7 fixed targets) and doesn't need a learned policy for v1. |
| Orchestration | **ROS2 Humble on Ubuntu 22.04** | Aligns with the "learn ROS2" goal, matches Hiwonder's official image, future-proof for MoveIt2/Gazebo/Nav2. |
| Arm control library | **Hiwonder Python SDK + ROS2 wrapper** | Use Hiwonder's bus servo SDK underneath a custom ROS2 action server. |

### Decisions explicitly not taken

- **Not buying:** SO-ARM101 / LeRobot kit. Considered for the leader/follower teleop paradigm, but the JetArm covers the v1 task more cleanly at the same price point and the LeRobot ecosystem isn't the goal.
- **Not buying:** xArm Lite 6 / Niryo Ned2. Better arms in every respect but blow the $1.5K budget.
- **Not building:** A custom riser. JetArm's integrated tall base eliminates the need.
- **Not 3D-printing:** Custom TPU gripper fingertips for v1. The stock HTS-21H gripper jaws will be evaluated first; revisit only if reliability falls short.

---

## 3. Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                 Jetson Orin Nano 8GB (Ubuntu 22.04 + ROS Humble) │
│                                                                  │
│  ┌──────────────┐   ┌──────────────┐   ┌─────────────────────┐  │
│  │ vision_node  │──▶│  game_node   │──▶│   policy_node       │  │
│  │ (existing)   │   │ (existing,   │   │   (AlphaZero ONNX,  │  │
│  │ RGB+D from   │   │  port from   │   │    existing model)  │  │
│  │ Gemini Plus  │   │  web/src/    │   │                     │  │
│  │              │   │  game/       │   │                     │  │
│  └──────┬───────┘   │  connectFour │   └──────────┬──────────┘  │
│         │           │  .js)        │              │             │
│         │           └──────────────┘              │             │
│         │                                         ▼             │
│         │           ┌─────────────────────────────────────┐     │
│         │           │       orchestrator_node             │     │
│         └──────────▶│  (game state machine, turn          │     │
│                     │   detection, move dispatch)         │     │
│                     └────────────────────┬────────────────┘     │
│                                          │                      │
│                                          │ /place_piece         │
│                                          │   (column N)         │
│                                          ▼                      │
│                     ┌─────────────────────────────────────┐     │
│                     │           arm_node                  │     │
│                     │  ROS2 action server                 │     │
│                     │  PlacePiece.action                  │     │
│                     │  ─ wraps Hiwonder Python SDK        │     │
│                     │  ─ executes scripted trajectories   │     │
│                     │  ─ reports success/failure          │     │
│                     └────────────────┬────────────────────┘     │
└──────────────────────────────────────┼──────────────────────────┘
                                       │ USB (STM32F407 controller)
                                       ▼
                       ┌──────────────────────────────────┐
                       │     Hiwonder JetArm hardware     │
                       │  ─ HTD-35H ×3 (body)             │
                       │  ─ HX-12H (wrist)                │
                       │  ─ HTS-21H (gripper)             │
                       │  ─ HTS-35H (camera pan-tilt)     │
                       │  ─ Orbbec Gemini Plus depth cam  │
                       └──────────────────────────────────┘
```

### Node responsibilities

| Node | Status | Owner |
|---|---|---|
| `vision_node` | Existing pipeline (RGB-only); upgrade to RGB+D from Gemini Plus | Reuse + extend |
| `game_node` | Port `web/src/game/connectFour.js` rules engine to Python | New (port) |
| `policy_node` | Wraps `ai/src/connect_four_ai/models/alphazero-network-model.onnx` via onnxruntime-gpu | Reuse |
| `orchestrator_node` | State machine: detect human turn end → invoke policy → dispatch to arm | New |
| `arm_node` | Action server exposing `PlacePiece.action`; wraps Hiwonder SDK underneath | New |

### ROS2 action contract

```
# PlacePiece.action
# Goal
int8 column           # 0..6
string color          # "red" | "yellow" (informational)
---
# Result
bool success
string error_code     # "" | "grasp_failed" | "drop_failed" | "timeout" | ...
---
# Feedback
string phase          # "moving_to_tray" | "grasping" | "lifting" | "moving_to_column" | "releasing" | "returning_home"
float32 progress      # 0.0..1.0
```

This is the only contract between the existing game/AI software and the new robot side. Anything below this line can be replaced (Hiwonder SDK → MoveIt2 → learned policy) without touching the AI or vision code.

---

## 4. Hardware

### Shopping list

| Item | Vendor | Status | Cost |
|---|---|---|---|
| JetArm Ultimate Kit (without Jetson) | Hiwonder via Amazon | Pending — awaiting Hiwonder support replies | $680 |
| Jetson Orin Nano 8GB Super Dev Kit | TBD | Ordering now | $249 |
| 256GB NVMe SSD | TBD | Ordering now | $75 |
| Misc cables, adapters, monitor cable | TBD | As needed | ~$50 |
| Tax + shipping | — | — | ~$80 |
| **Total** | | | **~$1,134** |

Comfortably under the $1,500 ceiling.

### What ships included with the JetArm Ultimate Kit (verify before ordering)

Confirmed via spec sheet:
- Aluminum-frame 6-servo arm with HTS-21H gripper, HX-12H wrist, HTD-35H ×3 body, HTS-35H pan-tilt
- Power adapter (12V 5A or 19V 2.37A)
- 5A high-current USB hub on base
- STM32F407VE16 low-level controller with FreeRTOS
- Full development manuals, video tutorials, ROS/STM32 source, system images
- Full-metal aluminum alloy frame, anodized, total height 532 mm

Confirmed for Ultimate variant specifically (vs Standard/Starter):
- Orbbec Gemini Plus 3D depth camera
- Built-in 6-microphone array
- 7-inch HD touchscreen
- Wireless RC transmitter / gamepad

### JetArm hard specs (from manufacturer spec sheet)

| Spec | Value |
|---|---|
| Dimensions | 339 × 165 × 532 mm |
| Weight | ~2.4 kg |
| Body material | Full-metal aluminum alloy, anodized |
| Low-level controller | STM32F407VE16 + FreeRTOS |
| OS image | Ubuntu 22.04 LTS + ROS Humble (or Ubuntu 20.04 + ROS Noetic) |
| Power | 12V 5A or 19V 2.37A |
| Servos (body) | 3 × HTD-35H, 35 kg·cm torque |
| Servos (wrist) | 1 × HX-12H |
| Servos (gripper) | 1 × HTS-21H |
| Camera | Orbbec Gemini Plus depth |
| Storage (with our own Jetson) | 256GB NVMe SSD (added separately) |

---

## 5. Open Questions for Hiwonder Support

Block ordering until these are answered:

1. **Confirm exact Amazon ASIN / SKU for the Ultimate Kit *without Jetson*.** Some Amazon URLs returned by search routed to the Standard kit; need a clean confirmation that the listing being purchased is Ultimate (with Gemini Plus + 6-mic array + 7" touchscreen).
2. **Horizontal reach.** What is the JetArm's horizontal reach in mm? Need to confirm it covers a 26 cm wide Connect Four board with the arm base set off behind the board. Spec sheet doesn't list this.
3. **Gripper jaw maximum opening (mm).** Connect Four discs are ~40 mm OD × ~6 mm thick. Need to confirm the HTS-21H gripper can grasp them on the rim.
4. **Repeatability spec.** Tip repeatability under typical light load (<100 g)? Want to verify ±2-3 mm or better for column pitch tolerance.
5. **Compatibility with Jetson Orin Nano 8GB Super Dev Kit.** The spec sheet lists Orin Nano support. Confirm specifically the **Super** variant works with their provided system image and SDK.
6. **Power adapter inclusion.** Confirm power adapter is in the box for the without-Jetson SKU.

Update this section as answers come in.

---

## 6. Plan

Three phases. Phase A starts immediately with Jetson + SSD on hand. Phase B starts when the arm arrives. Phase C is post-MVP polish.

### Phase A — Jetson-first work (~1 week, while arm ships)

Bring up the compute side, port existing code, build the ROS2 skeleton, and integrate against a stub arm node so end-to-end testing minus motion is possible before hardware arrives.

#### A.1 Jetson bring-up
- [ ] Mount 256GB NVMe SSD in Jetson Orin Nano 8GB Super Dev Kit
- [ ] Flash JetPack 6.x (or download Hiwonder's Ubuntu 22.04 + ROS Humble image once arm is confirmed)
- [ ] First boot, network, SSH, Tailscale (or equivalent) for headless dev
- [ ] Install ROS2 Humble desktop and dev tools (colcon, rosdep, vcstool)
- [ ] Install Python deps: `onnxruntime-gpu`, `numpy`, `opencv-python`, `pyserial`
- [ ] Verify CUDA + TensorRT install with `nvidia-smi`, `dpkg -l | grep tensorrt`

#### A.2 Existing code on Jetson
- [ ] Clone `eithan/connect-four` repo onto Jetson
- [ ] Install Python deps for `ai/` (poetry env)
- [ ] Run `alphazero-network-model.onnx` via `onnxruntime-gpu` with CUDA EP; benchmark inference latency
- [ ] If latency matters: try TensorRT EP via onnxruntime, then compare to native TensorRT engine
- [ ] Port `web/src/game/connectFour.js` rules engine to Python (`game_node` core logic)
- [ ] Confirm existing Connect Four vision + game-detection code runs; identify what's RGB-only vs ready for RGB+D

#### A.3 ROS2 workspace skeleton
- [ ] Create `ros2_ws/src/connect_four_arm/` package (ament_python)
- [ ] Define `PlacePiece.action` interface in `ros2_ws/src/connect_four_msgs/`
- [ ] Implement `arm_node` as a stub: accept goals, publish feedback every 200 ms, sleep through fake motion phases, return success
- [ ] Implement `orchestrator_node`: subscribes to `/board_state`, runs game loop, calls `arm_node` action
- [ ] Implement `policy_node`: wraps ONNX inference; publishes chosen column on game state change
- [ ] Wire stub `vision_node`: publishes a synthetic `/board_state` from a fake game progression for testing
- [ ] Add `launch/connect_four.launch.py` that brings up all five nodes
- [ ] Add a basic README under `ros2_ws/` explaining build + launch

#### A.4 Stub end-to-end test
- [ ] With all nodes running and a synthetic board state, verify the orchestrator → policy → arm flow completes a fake game
- [ ] Replay scenarios via `ros2 bag` to validate failure paths (action timeout, illegal move, etc.)

#### A.5 Documentation prep for Phase B
- [ ] Read Hiwonder JetArm wiki: chapters 1, 7 (URDF), 10–12 (ROS2 control), 16 (URDF in ROS2)
- [ ] Sketch the geometry: where the arm base, board, and piece tray will sit on the desk
- [ ] Order any 3D-printed parts (piece tray, board pedestal if needed) with the Flashforge Adventurer 3

**Phase A success criterion:** A synthetic game runs end-to-end through all five ROS2 nodes on the Jetson, AlphaZero inference is verified working under onnxruntime-gpu, and the only thing missing is real motion + real vision.

---

### Phase B — Hardware integration (~1 week, once arm arrives)

Replace the stub arm node with real motion. Introduce real vision. Play a real game.

#### B.1 Unboxing + verification
- [ ] Assemble JetArm per Hiwonder docs (if any final assembly needed)
- [ ] Power on, connect to Jetson, run Hiwonder's gamepad teleop demo to verify all 6 servos respond
- [ ] Run Hiwonder's depth-camera viewer to verify Gemini Plus
- [ ] Note any DOA or calibration issues; contact support before proceeding

#### B.2 Geometry lock
- [ ] Place arm + Connect Four board + piece tray on a flat fixed surface
- [ ] Measure: arm base position, board position (front edge of column slots), piece tray position
- [ ] Use clamps or screws so positions don't drift between sessions
- [ ] Print a structured piece tray (7 separated pockets) on the Flashforge if not improvised
- [ ] Document all positions in `ros2_ws/src/connect_four_arm/config/geometry.yaml`

#### B.3 Scripted pick-and-place (no ROS2 yet)
- [ ] Using Hiwonder Python SDK directly, teleop the arm via gamepad to capture 8 keyframe joint positions:
  - `home`
  - `pickup_tray`
  - `drop_col_0` … `drop_col_6`
- [ ] Save keyframes to `geometry.yaml`
- [ ] Write a standalone script that loops: `home → pickup_tray → drop_col_N → home` for each N
- [ ] Tune motion speeds and acceleration for smoothness (jerk-limited interpolation via `ruckig` or MoveIt2 TOTG)
- [ ] **Success criterion:** ≥95% successful drops in correct column over 50 trials

#### B.4 Real `arm_node`
- [ ] Replace stub `arm_node` motion with calls to the Phase B.3 motion library
- [ ] Action interface (`PlacePiece.action`) stays unchanged — orchestrator code from Phase A doesn't need to change
- [ ] Add servo health/error reporting via STM32 telemetry (position error, current draw, temperature)
- [ ] Add timeouts and joint-limit pre-checks
- [ ] Add `/estop` topic that immediately stops motion and parks at home

#### B.5 Real `vision_node`
- [ ] Calibrate Orbbec Gemini Plus intrinsics + extrinsics to the locked board position
- [ ] Port existing Connect Four vision + game-state code to publish `/board_state`
- [ ] Optional: leverage depth channel for hand-presence detection (suppress board-state updates while a hand is in frame)

#### B.6 End-to-end games
- [ ] Full game loop with a human opponent
- [ ] Iterate on failure modes (failed grasp recovery, board misread retries)
- [ ] **Success criterion:** Robot completes ≥3 full games against a human without intervention

**Phase B success criterion:** Functional end-to-end Connect Four robot that plays games against humans. Demo-able.

---

### Phase C — Polish + stretch (optional, post-MVP)

Listed in rough priority order; pick what's useful.

- [ ] **MoveIt2 motion planning** replacing hand-tuned trajectories; better collision avoidance, smoother paths
- [ ] **URDF + Gazebo simulation** so future changes can be tested without hardware
- [ ] **Failure recovery polish** (re-grasp on failure, better error messages, startup self-test)
- [ ] **Demo polish** (small GUI, status display on the arm's 7" touchscreen, start-of-game ritual, etc.)
- [ ] **Behavior cloning experiment** — record gamepad-teleop demos, train a small ACT/Diffusion policy as an alternative arm controller; compare to scripted motion
- [ ] **3D-printed custom gripper jaws** if stock gripper proves unreliable
- [ ] **Voice control** ("your turn", "good game") via the included 6-mic array
- [ ] **Far stretch:** explore SO-ARM101 + leader/follower as a separate platform for VLA work; do not retrofit onto JetArm

---

## 7. Existing Code Reuse

From [`eithan/connect-four`](https://github.com/eithan/connect-four):

| Existing | Reuse strategy |
|---|---|
| `ai/src/connect_four_ai/models/alphazero-network-model.onnx` (+ `.onnx.data`) | Loaded directly by `policy_node` via `onnxruntime-gpu` (CUDA EP) on Jetson |
| `ai/` — AlphaZero training pipeline | Not deployed at runtime; remains the offline training rig |
| `web/src/game/connectFour.js` rules engine | Port to Python in `game_node` |
| Existing Connect Four vision / detection code (per current state) | Reused as the basis for `vision_node`, upgraded with Gemini Plus depth |
| `web/` React frontend | Untouched; can run alongside as a debug visualization if useful |

The robot side is *purely additive* to the existing repo. Nothing in `web/`, `ai/`, or the existing tests should need to change.

---

## 8. Repository Additions

Proposed structure for the new robot-side code under the existing repo root:

```
connect-four/
├── ai/                         # existing AlphaZero training (unchanged)
├── web/                        # existing React frontend (unchanged)
├── docs/                       # existing docs
│   └── ROBOT_PLAN.md          # ← this file
├── robot/                      # NEW: Python helpers, calibration scripts, Hiwonder SDK glue
│   ├── calibrate.py
│   ├── teleop_capture.py       # capture keyframe joint positions via gamepad
│   ├── motion.py               # ruckig-based trajectory generation
│   └── tests/
└── ros2_ws/                    # NEW: ROS2 workspace
    └── src/
        ├── connect_four_msgs/
        │   └── action/
        │       └── PlacePiece.action
        ├── connect_four_arm/
        │   ├── config/
        │   │   └── geometry.yaml
        │   ├── connect_four_arm/
        │   │   ├── arm_node.py
        │   │   ├── orchestrator_node.py
        │   │   ├── policy_node.py
        │   │   ├── game_node.py
        │   │   └── vision_node.py
        │   ├── launch/
        │   │   └── connect_four.launch.py
        │   ├── package.xml
        │   ├── setup.py
        │   └── README.md
        └── connect_four_bringup/
            └── launch/
                └── full_stack.launch.py
```

The user's existing folders (`robot/`, `ai/`, `ros2_ws/`) referenced earlier in planning fit this structure naturally.

---

## 9. Risks & Mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Horizontal reach insufficient for far columns of vertical board | Low–Medium | Verify with Hiwonder support before ordering (open Q #2). If marginal, lay board flat or lift it on a small pedestal. |
| Stock gripper can't reliably grasp 4 cm disc | Low–Medium | Print custom TPU jaws (Phase C). Stock gripper is a parallel-jaw with adjustable opening; should be fine. |
| Hiwonder SDK has rough edges / poor docs | Medium | Their wiki and tutorials look comprehensive; community is smaller than LeRobot but active. Time-box debugging; fall back to direct STM32 serial if needed. |
| Choppy motion harms demo polish | Medium | Use jerk-limited trajectories (`ruckig`) or MoveIt2 TOTG. Tune speeds. JetArm's metal frame is a structural improvement over hobbyist arms. |
| ONNX inference too slow on Jetson | Low | Tiny AlphaZero net runs in <1ms expected. If MCTS rollouts are slow, that's CPU-bound and unaffected by GPU. |
| Sub-week 2 hardware shipping | Medium | Phase A is structured to absorb 1–3 weeks of waiting without idle time. |

---

## 10. References & Sources

- [`eithan/connect-four`](https://github.com/eithan/connect-four) — the existing repo this plan extends
- [Hiwonder JetArm product page](https://www.hiwonder.com/products/jetarm)
- [Hiwonder JetArm wiki — Getting Ready](https://wiki.hiwonder.com/projects/JetArm/en/jetarm-orin-nano/docs/1.Getting_Ready.html)
- [Hiwonder JetArm wiki — ROS2 Robot Arm Basic Control](https://wiki.hiwonder.com/projects/JetArm/en/jetarm-jetson-nano/docs/10.ROS2_Robot_Arm_Basic_ControlUser_Manual.html)
- [Hiwonder JetArm wiki — URDF Modeling & Simulation](https://docs.hiwonder.com/projects/JetArm/en/jetarm-orin-nano/docs/7.ROS1_Robot_Arm_URDF_Modeling_Simulation.html)
- [Hiwonder HTD-35H bus servo](https://www.hiwonder.com/products/htd-35h)
- [JetArm Ultimate Kit (without Jetson) — Amazon listing](https://www.amazon.com/Hiwonder-Robotic-Recognition-Education-Scenarios%EF%BC%88Standard/dp/B0D83X7Z1F)
- [NVIDIA Jetson Orin Nano Super Developer Kit](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/nano-super-developer-kit/)
- [ROS 2 Humble docs](https://docs.ros.org/en/humble/)
- [ruckig — jerk-limited trajectory generation](https://github.com/pantor/ruckig)
- [MoveIt2 docs](https://moveit.picknik.ai/main/index.html)
- [onnxruntime CUDA execution provider](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)

---

## 11. Changelog

- **2026-04-28** — Initial version. Decision locked on JetArm Ultimate (without Jetson). Three-phase plan with Phase A (Jetson-first) bringup while arm ships. Budget at ~$1,134 including 256GB NVMe SSD addition.
