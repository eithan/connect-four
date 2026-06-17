# Session Status

**Read this first.** A concise dashboard of the project's current state, hardware logistics, and immediate next steps. The full master plan lives in [`PROJECT_PLAN.md`](./PROJECT_PLAN.md).

**Last updated:** 2026-06-17

---

## Where We Are

| Area | Status |
|---|---|
| Phase 1 — Vision pipeline (static images) | ✅ Complete |
| Phase 2 — Live camera + cooperative game loop | ✅ Mostly complete (sub-tasks 2.6 / 2.7 deprioritized to Phase 5) |
| Phase 3 — ROS2 simulation core (UR5e + Gazebo + MoveIt2) | ✅ Complete |
| Phase 3B — VLA-ready sim (Robotiq 2F-85 + pick-and-place) | ✅ Algorithm-complete (grasp contact physics deferred to real hardware) |
| Phase 3C — SO-101 hardware setup | 🔄 25-episode col3 dataset merged. ACT training on Mac M4 Max (MPS) — 50k steps, ~loss 0.052 at step 14k, finishing ~1pm 2026-06-17. Ubuntu dropped; all recording, training, and inference now on Mac. |
| Phase 4 — Full game integration on real arm | 🔲 Pending |

## Decisions Locked

| Decision | Choice | Detail |
|---|---|---|
| Robot arm | **Hiwonder LeRobot SO-ARM101 Advanced Kit (assembled)** — leader + follower + both cameras | [`ARM_DECISION_LOG.md`](./ARM_DECISION_LOG.md) |
| Compute | **Mac M4 Max 64GB** for all recording, training, and inference | Ubuntu dropped — AMD GPU not ROCm-compatible, CPU inference too slow (2.3 Hz). Mac MPS runs inference at usable speed. |
| Jetson Orin Nano | **Deferred to Phase 5** (untethered/edge inference) | Saves ~$340 now; buy when actually needed |
| Sim arm | **UR5e + Robotiq 2F-85** (parallel research vehicle, validates algorithms) | Existing `connect_four_arm` ROS2 package |
| Gripper standard | **Robotiq 2F-85** (sim) → SO-101 stock parallel-jaw (real) | — |
| Software approach | ACT for single-column baseline → SmolVLA for multi-column generalization | See Training Roadmap below |
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
2. ✅ **Recording pipeline fully debugged** (2026-06-03).
3. ✅ **25-episode col3 dataset recorded and merged** (`eithanz/c4_col3_25`). 5 batches of 5 episodes merged with `lerobot-edit-dataset`.
4. ✅ **Ubuntu dropped as compute platform** (2026-06-17). AMD GPU not ROCm-compatible. All work now on Mac M4 Max.
5. 🔄 **ACT training in progress on Mac** (2026-06-17): 50k steps, loss ~0.052 at step 14k, finishing ~1pm today. Checkpoint will be at `outputs/train/act_c4_col3/checkpoints/last/`.
6. 🔲 **Next: deploy and test on arm** — see Rollout on Mac section below.

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

## Training & Deployment Roadmap

### Phase 1 — Validate current model (today, ~1–2 hours active)
Training finishes ~1pm 2026-06-17. Find serial port (`ls /dev/tty.usb*`) and camera indices (`lerobot-find-cameras` or use LeLab values: workspace_cam=0, wrist_cam=1). Run 10–20 attempts on the arm. Establish real-world baseline on col3, yellow pieces.

Decision point: below 40% = diagnose setup (calibration, camera pose). Above 40% = proceed.

### Phase 2 — Strengthen single-column baseline (~3–4 days)
Record 25 more yellow col3 episodes (~2 hours across a couple of sessions). Merge into 50-episode dataset (one-shot merge command, 10 mins). Retrain ACT overnight (~21 hours). Test again. Expected: 75–85% success rate.

### Phase 3 — Build multi-column VLA dataset (~2–3 weeks of recording)
Record 15 episodes per column, all 7 columns, yellow first (105 episodes). At 5 per session (~20–30 mins each) = ~21 sessions over 2 weeks. Then 10 episodes per column in red (70 more episodes, ~1 week). Roll each batch of 5 into master with `merge_datasets.sh` as you go. Total: ~175 episodes, ~15–20 hours active recording.

### Phase 4 — Fine-tune SmolVLA (~1 day, mostly automated)
SmolVLA (~450M params) is too slow for MPS — use a cloud GPU (Lambda Labs / RunPod A100: 2–4 hours, ~$5–10). Result: one model handling all 7 columns and both colors from natural language prompts.

### Phase 5 — Game integration (scope TBD)
Connect vision system (board state) to VLA policy. Game logic picks target column, passes to VLA as natural language instruction, arm executes. Software integration more than robotics — plan in detail once Phase 4 is solid.

---

## Rollout on Mac

```bash
CKPT="$HOME/development/cursor/connect-four/robot/outputs/train/act_c4_col3/checkpoints/last/pretrained_model"

PYTORCH_ENABLE_MPS_FALLBACK=1 lerobot-rollout \
  --robot.type=so101_follower \
  --robot.port=/dev/tty.usbmodem5C4C1287431 \
  --robot.id=my_follower_arm \
  --robot.cameras='{
    workspace_cam: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30},
    wrist_cam:     {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}
  }' \
  --policy.type=act \
  --policy.pretrained_path="${CKPT}" \
  --device=mps \
  --fps=30 \
  --task="Pick a yellow piece from the tray and drop it into column 0" \
  --display_data=false \
  --strategy.type=base
```

**Notes:**
- SO101_LEADER: `/dev/tty.usbmodem5C4C1276591`
- SO101_FOLLOWER: `/dev/tty.usbmodem5C4C1287431`
- CAM_WORKSPACE: 0, CAM_WRIST: 1
- `--fps=30` must match dataset recording fps
- `PYTORCH_ENABLE_MPS_FALLBACK=1` required — some ops not natively supported on MPS
- Known Ctrl-C disconnect error (motor id=3 on teardown): non-dangerous. Workaround: `--return_to_initial_position=false`

## Changelog

- **2026-06-17** — Ubuntu dropped as compute platform. All recording, training, and inference now on Mac M4 Max 64GB. Recorded 25 episodes across 5 batches (col3, yellow), merged into `eithanz/c4_col3_25` with `lerobot-edit-dataset`. ACT training in progress: 50k steps, loss ~0.052 at step 14k, finishing ~1pm. Added Training & Deployment Roadmap (5 phases: validate → strengthen baseline → multi-column VLA dataset → SmolVLA fine-tune → game integration). Fixed rollout command: camera keys updated to `workspace_cam`/`wrist_cam`, fps corrected to 30, device to mps. Port and camera indices confirmed: SO101_LEADER=`/dev/tty.usbmodem5C4C1276591`, SO101_FOLLOWER=`/dev/tty.usbmodem5C4C1287431`, CAM_WORKSPACE=0, CAM_WRIST=1.

- **2026-06-07** — ACT rollout pipeline fully debugged on Ubuntu. Fixed `context.py` `from_pretrained` config-override bug (empty `input_features` from CLI overrode checkpoint's `config.json`); fixed checkpoint `config.json` device mismatch (mps→cpu for Ubuntu). Policy loads and runs. CPU-only inference at 2.3–2.5 Hz vs 15 Hz required — arm misses. AMD RX 460D not supported by ROCm 6.x. `torch.compile` fails silently on this build. Next path: run rollout on Mac M4 Max with MPS. Mac checkpoint restored from accidentally-deleted outputs folder (all checkpoints 010k–050k sorted, timestamp suffixes stripped); 050000/pretrained_model/model.safetensors still needs rsync from Ubuntu.

- **2026-06-03** — Recording pipeline fully debugged. Resolved: lerobot resume root-path inconsistency (create vs resume use different root values), HF Hub API call on resume with push_to_hub=false, camera labels swapped (video0↔video2), FPS warnings fixed by dropping to 15fps + streaming encoding + display_data=false, AV1 playback requires mpv/ffplay not VLC, arrow key injection via xdotool (pynput uses X11 backend on RDP, not evdev), Ctrl-C timing for safe episode saves. Camera view finalized: overhead rear behind board.

- **2026-05-22** — SO-101 **arrived early** (well ahead of the early-June ETA), assembled, calibrated, and teleoperating with both cameras (scene `/dev/video0` + wrist `/dev/video2`) in the Rerun viewer. Phase 3C advanced from pre-arm prep to demo recording. Added the `connect_four_piece_edge_nest.scad` fixture (holds a disc on edge for a vertical face-grip, so the disc drops straight into the board — supersedes the flat rim-grip approach in the tray/chute/magazine files for v1, and sidesteps the parked wrist-rotation question). Added `dataset_schema.md` and `record_first_dataset.sh` for the v1 fixed-pick → column-0 dataset. Discs confirmed at 32 mm × 8.5 mm.
- **2026-04-30** — SO-101 Advanced Kit ordered from Hiwonder ($540 total, ~25 business days from China, ETA early June 2026). Phase 3B flipped to ✅ ALGORITHM-COMPLETE in `PROJECT_PLAN.md` with closeout note documenting the parked grasp-physics tuning. Compute decision locked: defer Jetson Orin Nano to Phase 5, use existing Ubuntu 24.04 machine for Phase 3C/4 — LeRobot's canonical workflow, saves ~$340. Added prioritized TODO section for prep work during the ~5-week wait. User pacing: starting TODO items the week of 2026-05-04, slow tempo.
- **2026-04-29** — Restructured the doc set: renamed `ROBOT_PLAN.md` → `ARM_DECISION_LOG.md` (decision history) and `ROBOT_PLAN_ALT.md` → `VLA_FINETUNING_PLAN.md` (focused VLA companion to the master plan). Added this `SESSION_STATUS.md` as the dashboard entry point. Folded the Physical Intelligence architecture insights into `PROJECT_PLAN.md` and `VLA_FINETUNING_PLAN.md`.
- **2026-04-28** — JetArm evaluation closed; SO-101 decision locked. Hiwonder support replies received and recorded in `ARM_DECISION_LOG.md`.
