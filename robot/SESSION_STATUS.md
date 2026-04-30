# Session Status

**Read this first.** A concise dashboard of the project's current state, hardware logistics, and immediate next steps. The full master plan lives in [`PROJECT_PLAN.md`](./PROJECT_PLAN.md).

**Last updated:** 2026-04-29

---

## Where We Are

| Area | Status |
|---|---|
| Phase 1 — Vision pipeline (static images) | ✅ Complete |
| Phase 2 — Live camera + cooperative game loop | ✅ Complete |
| Phase 3 — ROS2 simulation core (UR5e + Gazebo + MoveIt2) | ✅ Complete |
| Phase 3B — VLA-ready sim (Robotiq 2F-85 + pick-and-place) | 🔄 In progress, mid-tuning |
| Phase 3C — SO-101 hardware setup | 🔲 Next, blocked on hardware shipping |
| Phase 4 — Full game integration on real arm | 🔲 Pending |

## Decisions Locked

| Decision | Choice | Detail |
|---|---|---|
| Robot arm | **LeRobot SO-ARM101** (leader + follower kit, fully assembled) | [`ARM_DECISION_LOG.md`](./ARM_DECISION_LOG.md) |
| Compute | **Jetson Orin Nano 8GB Super Dev Kit** | — |
| Storage | **256GB NVMe SSD** | Sufficient for this project; 1 TB if doing heavy VLA work |
| Sim arm | **UR5e + Robotiq 2F-85** (parallel research vehicle, validates algorithms) | Existing `connect_four_arm` ROS2 package |
| OS / ROS | **Ubuntu 24.04 + ROS2 Jazzy** | Existing Ubuntu runtime machine |
| Gripper standard | **Robotiq 2F-85** (sim) → SO-101 stock parallel-jaw (real) | — |
| Software approach | Hybrid: scripted arm motion + learned vision/AI for v1; ACT then SmolVLA for v2 | — |
| Long-term direction | LeRobot ecosystem; openpi-compatible LeRobot dataset format from day one | [`VLA_FINETUNING_PLAN.md`](./VLA_FINETUNING_PLAN.md) |

## Hardware Logistics

| Item | Status | Cost (USD) | Notes |
|---|---|---|---|
| Jetson Orin Nano 8GB Super Dev Kit | Ordering now | $250 | — |
| 256GB NVMe SSD | Ordering now | $75 | Single M.2 slot on Orin Nano carrier |
| LeRobot SO-ARM101 (leader + follower, assembled) | Pending order | ~$470 | Seeed Studio / Hiwonder / WowRobo |
| Logitech C920 (overhead camera) | Pending | ~$70 | Or use existing if available |
| USB endoscope cam (wrist) | Pending | ~$30 | — |
| Cables, USB hub, misc | Pending | ~$50 | — |
| Filament + hardware (riser / fingertips / tray) | On hand | — | Flashforge Adventurer 3 |
| **Total** | | **~$945** | Comfortably under $1,500 ceiling |

Hardware lead time estimate: 1–3 weeks for SO-101 (varies by vendor). Phase A pre-arm work absorbs the wait.

## Current Focus

1. **Finish Phase 3B tuning** — verify wrist rotation direction in `column_mover.py`, finalize `GRIPPER_PIECE` value for the 38 mm Hasbro disc, confirm pick-and-place reliability across all 7 columns in Gazebo.
2. **Order SO-101 leader+follower kit** — JetArm evaluation closed (see `ARM_DECISION_LOG.md`). Vendor choice TBD between Seeed Studio, Hiwonder, and WowRobo based on availability and shipping time.
3. **Pre-arm prep on the Ubuntu machine** while waiting for SO-101 to arrive:
   - Install LeRobot v0.5.0+ (`pip install -e ".[feetech]"` from HuggingFace fork)
   - Read LeRobot docs front-to-back: teleop, dataset recording, ACT/SmolVLA training
   - Skim `Physical-Intelligence/openpi` GitHub for fine-tuning workflow

## Immediate Next Steps

- [ ] Place Jetson Orin Nano 8GB Super + 256GB NVMe order
- [ ] Decide SO-101 vendor (Seeed / Hiwonder / WowRobo) and place order
- [ ] Phase 3B: confirm wrist_1 is the correct rotation joint; tune `GRIPPER_PIECE` for 38 mm piece
- [ ] Install LeRobot environment on Ubuntu machine
- [ ] (Optional) CAD the SO-101 riser, piece tray, and TPU fingertips on the Flashforge while waiting for arm

## Open Questions

- Will Hiwonder RMA the JetArm Ultimate Kit if the Jetson Orin Nano *Super* specifically doesn't work? (Their reply was "should be compatible," not officially tested.) — Note: only relevant if we ever revisit the JetArm decision; currently moot since SO-101 is locked.
- Best vendor for SO-101 leader+follower assembled kit (price, lead time, gripper variant).
- Will we attempt to add SO-101 URDF to the Gazebo simulation, or keep UR5e as the sim platform and SO-101 as real-only? (Per `PROJECT_PLAN.md` Phase 3C, real demos beat sim demos for VLA training, so the answer is probably "real-only, keep UR5e in sim as algorithm validation.")

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
| `robot/ai_player.py` | AlphaZero ONNX inference + heuristic fallback |
| `robot/turn_tracker.py` | Stateful turn detection / win detection |
| `ros2_ws/src/connect_four_arm/scripts/column_mover.py` | Arm control node (IK precompute + joint-space planning) |
| `ros2_ws/src/connect_four_arm/launch/connect_four.launch.py` | Single-command full stack launch |
| `ai/src/connect_four_ai/models/alphazero-network-model.onnx` | Trained AlphaZero model (also at `robot/alphazero-network-model.onnx`) |

## Changelog

- **2026-04-29** — Restructured the doc set: renamed `ROBOT_PLAN.md` → `ARM_DECISION_LOG.md` (decision history) and `ROBOT_PLAN_ALT.md` → `VLA_FINETUNING_PLAN.md` (focused VLA companion to the master plan). Added this `SESSION_STATUS.md` as the dashboard entry point. Folded the Physical Intelligence architecture insights into `PROJECT_PLAN.md` and `VLA_FINETUNING_PLAN.md`.
- **2026-04-28** — JetArm evaluation closed; SO-101 decision locked. Hiwonder support replies received and recorded in `ARM_DECISION_LOG.md`.
