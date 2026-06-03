# Session Status

**Read this first.** A concise dashboard of the project's current state, hardware logistics, and immediate next steps. The full master plan lives in [`PROJECT_PLAN.md`](./PROJECT_PLAN.md).

**Last updated:** 2026-06-03

---

## Where We Are

| Area | Status |
|---|---|
| Phase 1 — Vision pipeline (static images) | ✅ Complete |
| Phase 2 — Live camera + cooperative game loop | ✅ Mostly complete (sub-tasks 2.6 / 2.7 deprioritized to Phase 5) |
| Phase 3 — ROS2 simulation core (UR5e + Gazebo + MoveIt2) | ✅ Complete |
| Phase 3B — VLA-ready sim (Robotiq 2F-85 + pick-and-place) | ✅ Algorithm-complete (grasp contact physics deferred to real hardware) |
| Phase 3C — SO-101 hardware setup | 🔄 Recording pipeline fully debugged (2026-06-03). Ready to collect the 50-episode v1 dataset. |
| Phase 4 — Full game integration on real arm | 🔲 Pending |

## Decisions Locked

| Decision | Choice | Detail |
|---|---|---|
| Robot arm | **Hiwonder LeRobot SO-ARM101 Advanced Kit (assembled)** — leader + follower + both cameras | [`ARM_DECISION_LOG.md`](./ARM_DECISION_LOG.md) |
| Initial compute | **Existing Ubuntu 24.04 machine** controls the arm directly via USB | LeRobot canonical workflow; no Jetson needed for Phase 3C/4 |
| Jetson Orin Nano | **Deferred to Phase 5** (untethered/edge inference) | Saves ~$340 now; buy when actually needed |
| Sim arm | **UR5e + Robotiq 2F-85** (parallel research vehicle, validates algorithms) | Existing `connect_four_arm` ROS2 package |
| OS / ROS | **Ubuntu 24.04 + ROS2 Jazzy** | Existing Ubuntu runtime machine |
| Gripper standard | **Robotiq 2F-85** (sim) → SO-101 stock parallel-jaw (real) | — |
| Software approach | Hybrid: scripted arm motion + learned vision/AI for v1; ACT then SmolVLA for v2 | — |
| Long-term direction | LeRobot ecosystem; openpi-compatible LeRobot dataset format from day one | [`VLA_FINETUNING_PLAN.md`](./VLA_FINETUNING_PLAN.md) |

## Hardware Logistics

### Active purchase: SO-101 Advanced Kit

| Item | Status | Cost (USD) | Notes |
|---|---|---|---|
| Hiwonder SO-ARM101 Advanced Kit (assembled) | ✅ Ordered 2026-04-30 | $460 | Includes leader + follower arms + wrist camera + external scene camera + all cables + power supplies + BusLinker V3.0 debugging board |
| Shipping | — | $80 | Shipping from China |
| Powered USB hub | Pending | ~$30 | If Linux machine needs more ports for arms + cameras |
| Spare USB-C cable | Pending | ~$10 | Backup |
| Filament + hardware (riser / fingertips / tray) | On hand | — | Flashforge Adventurer 3 |
| **Active total** | | **~$580** | Well under the $1,500 ceiling |

**Lead time:** ~25 business days (~5 calendar weeks) from China. Expected arrival window: early June 2026. Pre-arm prep work absorbs the wait — see TODO section below. The user has indicated they will start the TODO items the week of 2026-05-04 and pace them slowly.

### Deferred to Phase 5 (untethered/edge inference)

| Item | Cost (USD) | Trigger to reactivate |
|---|---|---|
| Jetson Orin Nano 8GB Super Dev Kit | $250 | When you want the robot to run standalone without a host PC |
| 256GB NVMe SSD | $75 | Bought together with Jetson |
| **Deferred total** | **~$340** | Saves ~$340 now; reactivates when Phase 5 begins |

Reasoning: LeRobot's canonical workflow connects the SO-101 leader + follower over USB to whatever machine runs LeRobot Python. The existing Ubuntu 24.04 machine already runs ROS2 Jazzy + Gazebo and is well-suited for Phase 3C (calibration, demo recording, ACT training) and Phase 4 (full game integration). The Jetson only becomes useful in Phase 5 when on-arm/edge inference is needed for untethered operation. Buying it now would put it on a shelf for months.

## Current Focus

1. ✅ **SO-101 arrived, assembled, calibrated, teleop verified** (~2026-05-22).
2. ✅ **Recording pipeline fully debugged** (2026-06-03). All major issues resolved — see Recording Setup section below.
3. 🔄 **Next: collect 50-episode v1 dataset.** Run `record_first_dataset.sh`, do the full pick-from-chute → drop-column-0 task. Once 50 episodes are done, run `train_act.sh`.

## Recording Setup (as of 2026-06-03)

Everything needed to start recording is in place. Key facts:

| Item | Detail |
|---|---|
| Script | `robot/record_first_dataset.sh` |
| Dataset path | `~/lerobot_datasets/eithanz/connect_four_chute5_pick_col0/` |
| FPS | 15 (30 caused consistent frame drops due to YUYV wrist cam + AV1 encoding overhead) |
| Cameras | `front`=/dev/video2 (overhead rear, looking down at workspace); `hand`=/dev/video0 (wrist) |
| Wrist cam format | YUYV (can't do MJPG — falls back silently) |
| Encoding | Streaming AV1 (libsvtav1); use `mpv` or `ffplay` to play back — VLC 3.x can't decode AV1 |
| Arrow keys | Use `python3 robot/send_key.py` in a second terminal (xdotool-based, requires `sudo apt install xdotool`) |
| Ctrl-C safety | Wait until log shows `Recording episode N` (not `Reset the environment`) before Ctrl-C — episode data writes during the reset phase |
| Resuming | Script auto-detects existing dataset and passes `--resume=true` with correct root path |
| Viewing episodes | `./view_episode.sh 0` (uses ffplay, correct chunk/file path format) |

**Camera position:** overhead rear — camera mounted behind and above the Connect Four board, looking down toward the arm and workspace. Full arm in frame, board back-face visible on right side. Static shade visible upper-left (harmless). Monitor not in frame.

**Board state between episodes:** fine to leave pieces in the board — the task trajectory is the same regardless of board state. Clear column 0 only when it fills up.

## TODO: Prep Work While Waiting for Arm

Prioritized so nothing is blocking when the arm arrives. Items 1–3 are minimum-viable prep; the rest is bonus.

### 1. Install and smoke-test the LeRobot environment

Non-trivial install with several gotchas (CUDA versions, Feetech serial drivers, pyenv conflicts). Do this now so it's ready on day 1.

- [ ] `git clone https://github.com/huggingface/lerobot.git && cd lerobot && pip install -e ".[feetech]"`
- [ ] Run `python lerobot/scripts/find_motors_bus_port.py` — will fail without arm, that's expected; we're verifying the environment, not the hardware
- [ ] Pull SmolVLA and pi0_base checkpoints from HuggingFace Hub so they're cached locally
- [ ] Read the LeRobot tutorial notebooks end-to-end: teleop, recording, ACT training, evaluation
- [ ] Confirm CUDA + GPU works for ACT training on the Ubuntu machine (`nvidia-smi`, run a small test forward pass)

### 2. Design and 3D-print the workspace fixtures

You have the Flashforge and time. Print:

- [ ] **Piece tray** with 7–14 structured pockets sized for ~40 mm × 7 mm discs. Pockets should chamfer slightly so the gripper self-centers on pickup.
- [ ] **Arm riser** if your bench geometry needs one. Measure your Connect Four board height vs. the SO-101's reach envelope (429 mm horizontal, ~340 mm vertical).
- [ ] **Cable management plate** (optional) — bolts to the back of the SO-101 base, routes USB and power cleanly.
- [ ] **Defer:** custom TPU fingertips. Wait until the arm arrives so you can measure actual jaw geometry before printing.

### 3. Order the remaining consumables

Don't have these arrive a week after the arm.

- [ ] Powered USB hub if the Ubuntu machine doesn't have enough ports for arm controllers + cameras (likely — count: 2 for arms + 2 for cameras + anything else)
- [ ] Spare USB-C cable
- [ ] M3 brass heat-set inserts and screws (for the riser/fixtures)
- [ ] Confirm Ubuntu machine has at least 4 free USB ports total (2 arm controllers + 2 cameras) — otherwise add a powered hub

### 4. Design the `arm_node` architecture for the SO-101

Design only — no code can be smoke-tested against hardware yet. Decide:

- [ ] **Topic interface:** Keep the existing `/connect_four/pick_and_place` (String, "color,col") so `game_loop.py` doesn't change, OR switch to a typed action interface. Recommendation: keep the existing topic since it's already wired up.
- [ ] **Underlying motion library:** LeRobot Python `record`/`replay` for v1 scripted motion, with a path to swap in an ACT/SmolVLA policy server in v2.
- [ ] **Failure handling:** Servo torque/current proxy for "failed grasp" detection, timeouts per phase, recovery to home pose.
- [ ] **Sketch the ROS2 node in Python**, even if it won't be testable until the arm arrives.

### 5. Define the VLA dataset annotation schema

Decide *now* what language strings will accompany each episode in the LeRobot dataset, before recording any demos. Write it down in a `dataset_schema.md` in `robot/`.

- [ ] Format: e.g., `"pick a {color} piece and place it in column {col}"` with `color ∈ {red, yellow}`, `col ∈ {0..6}`
- [ ] Episode boundaries: one pick-and-place = one episode? Or full game = one episode?
- [ ] Metadata schema: source tray slot, lighting condition, camera mounting, etc.

### 6. Read Physical Intelligence's openpi remote inference docs

Even if you start with SmolVLA on-machine, the option to add π₀.₅-base via remote inference is worth knowing the shape of.

- [ ] `Physical-Intelligence/openpi/blob/main/docs/remote_inference.md`
- [ ] The websocket protocol for action chunk streaming
- [ ] How to spin up a policy server with `serve_policy.py`

### 7. Phase 3B closeout

Don't pull the sim work — close it out cleanly so it stays useful as a sandbox.

- [ ] Add a comment to `column_mover.py` noting the wrist rotation direction question is parked
- [x] ~~Update `PROJECT_PLAN.md` Phase 3B status from "🔄 IN PROGRESS" to "✅ ALGORITHM-COMPLETE (grasp physics deferred to real hardware)"~~ — done 2026-04-30
- [ ] Commit any local sim tweaks so they're preserved

### 8. (Optional) Try a small demo recording on a sim SO-100

LeRobot supports a SO-100 in MuJoCo. Walking through the LeRobot record-train-deploy pipeline once with a simulated arm helps internalize the workflow before doing it for real. Skip if you'd rather not learn twice.

## Open Questions

- **Does the Ubuntu machine have a discrete GPU available for ACT training?** If yes, training is straightforward; if no, options are MPS on the Mac, cloud GPU rental, or accepting slow CPU training. Worth checking before TODO item 1.
- **How many free USB ports on the Ubuntu machine?** Need at least 4 (2 arm controllers + 2 cameras). If short, the powered USB hub on the consumables list becomes mandatory rather than optional. TODO item 3 covers this.
- **Will we add SO-101 URDF to the Gazebo simulation, or keep UR5e as the sim platform and SO-101 as real-only?** Per `PROJECT_PLAN.md` Phase 3C, real demos beat sim demos for VLA training, so the leaning answer is "real-only, keep UR5e in sim as algorithm validation." Decide before recording first demos.
- **When is the right time to revisit the Jetson decision?** Likely Phase 5 when untethered/edge deployment becomes a real goal.

## Key File Locations

| File | Purpose |
|---|---|
| `robot/PROJECT_PLAN.md` | Master plan, full phase breakdown |
| `robot/ARM_DECISION_LOG.md` | Why SO-101 over JetArm |
| `robot/VLA_FINETUNING_PLAN.md` | SmolVLA / π₀ / π₀.₅ deep-dive, openpi remote inference architecture |
| `robot/SESSION_STATUS.md` | This file — concise dashboard |
| `robot/CLAUDE.md` | Guidance for Claude when working in this folder |
| `robot/game_loop.py` | Live camera + cooperative game loop (with `--ros` flag) |
| `robot/board_detector.py` | Vision pipeline (HSV + YOLOv8 hybrid) |
| `robot/teleop.sh` | `lerobot-teleoperate` command (stable symlink ports, `front`/`hand` cameras) |
| `robot/connect_four_piece_edge_nest.scad` | Stage-1 edge-standing piece nest (vertical face-grip, drop-ready) |
| `robot/dataset_schema.md` | v1 dataset schema: task scope, language annotation, episode/metadata |
| `robot/record_first_dataset.sh` | `lerobot-record` command for the v1 fixed-pick → column-0 dataset |
| `robot/train_act.sh` | `lerobot-train` ACT command + commented real-arm eval/deploy command |
| `robot/ai_player.py` | AlphaZero ONNX inference + heuristic fallback |
| `robot/turn_tracker.py` | Stateful turn detection / win detection |
| `ros2_ws/src/connect_four_arm/scripts/column_mover.py` | Arm control node (IK precompute + joint-space planning) |
| `ros2_ws/src/connect_four_arm/launch/connect_four.launch.py` | Single-command full stack launch |
| `ai/src/connect_four_ai/models/alphazero-network-model.onnx` | Trained AlphaZero model (also at `robot/alphazero-network-model.onnx`) |

## Changelog

- **2026-06-03** — Recording pipeline fully debugged. Resolved: lerobot resume root-path inconsistency (create vs resume use different root values), HF Hub API call on resume with push_to_hub=false, camera labels swapped (video0↔video2), FPS warnings fixed by dropping to 15fps + streaming encoding + display_data=false, AV1 playback requires mpv/ffplay not VLC, arrow key injection via xdotool (pynput uses X11 backend on RDP, not evdev), Ctrl-C timing for safe episode saves. Camera view finalized: overhead rear behind board.

- **2026-05-22** — SO-101 **arrived early** (well ahead of the early-June ETA), assembled, calibrated, and teleoperating with both cameras (scene `/dev/video0` + wrist `/dev/video2`) in the Rerun viewer. Phase 3C advanced from pre-arm prep to demo recording. Added the `connect_four_piece_edge_nest.scad` fixture (holds a disc on edge for a vertical face-grip, so the disc drops straight into the board — supersedes the flat rim-grip approach in the tray/chute/magazine files for v1, and sidesteps the parked wrist-rotation question). Added `dataset_schema.md` and `record_first_dataset.sh` for the v1 fixed-pick → column-0 dataset. Discs confirmed at 32 mm × 8.5 mm.
- **2026-04-30** — SO-101 Advanced Kit ordered from Hiwonder ($540 total, ~25 business days from China, ETA early June 2026). Phase 3B flipped to ✅ ALGORITHM-COMPLETE in `PROJECT_PLAN.md` with closeout note documenting the parked grasp-physics tuning. Compute decision locked: defer Jetson Orin Nano to Phase 5, use existing Ubuntu 24.04 machine for Phase 3C/4 — LeRobot's canonical workflow, saves ~$340. Added prioritized TODO section for prep work during the ~5-week wait. User pacing: starting TODO items the week of 2026-05-04, slow tempo.
- **2026-04-29** — Restructured the doc set: renamed `ROBOT_PLAN.md` → `ARM_DECISION_LOG.md` (decision history) and `ROBOT_PLAN_ALT.md` → `VLA_FINETUNING_PLAN.md` (focused VLA companion to the master plan). Added this `SESSION_STATUS.md` as the dashboard entry point. Folded the Physical Intelligence architecture insights into `PROJECT_PLAN.md` and `VLA_FINETUNING_PLAN.md`.
- **2026-04-28** — JetArm evaluation closed; SO-101 decision locked. Hiwonder support replies received and recorded in `ARM_DECISION_LOG.md`.
