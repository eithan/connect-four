# Robot Arm Decision Log

**Status:** Closed. Decision: **LeRobot SO-ARM101** (leader + follower kit).
**See also:** [`PROJECT_PLAN.md`](./PROJECT_PLAN.md) for the active plan, [`VLA_FINETUNING_PLAN.md`](./VLA_FINETUNING_PLAN.md) for the VLA companion.

This document is a historical record of the arm-selection process. It captures the alternatives considered (notably the Hiwonder JetArm), the data gathered from manufacturers, and the reasoning that led to the SO-ARM101. Useful if the choice is ever revisited.

---

## 1. Final Decision

**LeRobot SO-ARM101 (leader + follower, fully assembled).** Decided 2026-04-28.

### Why SO-101 won

1. **Ecosystem alignment.** SO-101 is the canonical hardware target for the HuggingFace LeRobot stack and Physical Intelligence's openpi VLA fine-tuning recipes. SmolVLA was designed specifically for SO-100/101. JetArm has no equivalent path.
2. **Existing investment.** The repo already has a working ROS2 Jazzy + Gazebo Harmonic + MoveIt2 simulation built around UR5e + Robotiq 2F-85. Switching to JetArm would require abandoning that simulation and rebuilding URDF/SRDF/MoveIt configs around Hiwonder's HTS-21H gripper.
3. **Leader/follower teleoperation.** Built into the SO-101 kit; the canonical method for collecting imitation-learning demos. JetArm doesn't ship with a leader and the HTD-35H servos are stiff for hand-puppeting.
4. **Cross-embodiment generalization.** Skills built around the LeRobot toolchain transfer to Koch, Moss, ALOHA followers, and Trossen ViperX. JetArm-specific knowledge stays inside Hiwonder's ecosystem.
5. **Stated long-term goal: generalist robot teaching.** SO-101 sits on the canonical research progression (SO-101 → ALOHA-style bimanual → mobile manipulator). JetArm's next rung is unclear.

### What SO-101 costs

- Less polished out-of-box motion (compensated by `ruckig`-based jerk-limited trajectories + PID tuning)
- More fabrication (3D-printed riser, custom TPU fingertips, piece tray) — done on the Flashforge Adventurer 3
- ROS2 driver is community-maintained, not vendor-shipped (mitigated by wrapping LeRobot Python under a thin ROS2 action server)
- Reach is borderline at a vertical Hasbro board without a riser (mitigated by ~12 cm printed riser)

These costs are accepted as the curriculum, not waste — they're the substrate of "becoming someone who can teach robot arms generalist tasks."

---

## 2. Alternatives Considered

### Hiwonder JetArm Ultimate Kit (without Jetson) — strongest alternative

Considered through several rounds of evaluation. Technically capable, but rejected on ecosystem grounds.

**Pricing (verified):**
- JetArm Ultimate Kit (without Jetson): $680
- Jetson Orin Nano 8GB Super Dev Kit: $250
- 256GB NVMe SSD: $75
- Tax/shipping: ~$80
- **Total: ~$1,085** — comfortably under the $1,500 ceiling

**Hardware specs (manufacturer spec sheet):**

| Spec | Value |
|---|---|
| Dimensions | 339 × 165 × 532 mm |
| Weight | ~2.4 kg |
| Body material | Full-metal aluminum alloy, anodized |
| Servos (body) | 3 × HTD-35H (35 kg·cm torque) |
| Servos (wrist) | 1 × HX-12H |
| Servos (gripper) | 1 × HTS-21H |
| Pan-tilt camera | 1 × HTS-35H |
| Low-level controller | STM32F407VE16 + FreeRTOS |
| OS image | Ubuntu 22.04 LTS + ROS Humble |
| Power | 12V 5A or 19V 2.37A |
| Camera | Orbbec Gemini Plus depth |
| Bundled | 6-mic array, 7" touchscreen, wireless gamepad |

**Hiwonder support replies (confirmed):**

- **Compatibility with Jetson Orin Nano *Super*:** "Should be fully compatible. JetPack versions for both the Super and the standard Orin are quite similar." Not officially tested.
- **Working range:** 0.25–2.5 m (depth camera recognition range; arm reach is separate).
- **Maximum arm reach:** **429 mm horizontal.** Comfortably more than enough for a 26 cm wide Hasbro board.
- **Grasping capability:** Confirmed.
- **Precision/repeatability:** **2–4 mm.** Well within the ~13 mm of margin per Connect Four column slot.

**Pros:**
- Metal aluminum frame, industrial-bearing base, smoother out-of-box motion than SO-101
- 3D depth camera (Orbbec Gemini Plus) included
- Native ROS2 + MoveIt2 from Hiwonder; URDF and Gazebo simulation provided
- Tall integrated base (~53 cm) — no riser needed for vertical board
- All-in-one kit; minimal fabrication required
- Single-vendor support; comprehensive tutorials

**Cons (the reasons it lost):**
- Outside the LeRobot/openpi ecosystem; no SmolVLA / π₀ / π₀.₅ fine-tuning path
- No leader/follower teleoperation; HTD-35H servos are too stiff for hand-puppeting as makeshift leaders
- Switching from the existing UR5e + Robotiq 2F-85 + MoveIt2 simulation would require rebuilding the URDF/SRDF/MoveIt stack around HTS-21H gripper geometry
- Hiwonder ecosystem is more closed; smaller community for cross-embodiment transfer learning
- Joint count: 4-axis arm + 1-axis wrist + gripper (vs. SO-101's 5-axis + gripper) — slightly less wrist dexterity, though immaterial for Connect Four

### Other alternatives considered briefly

- **Elephant Robotics MyCobot 280-Pi** (~$700–900). 6-DOF metal arm, native ROS2 support. Smoother than SO-101 in most demos. Lost on the same ecosystem-alignment grounds as JetArm: not a first-class LeRobot citizen, no canonical VLA fine-tuning path.
- **Hiwonder ArmPi Ultra (RPi version)** (~$300–400). Cheaper. Similar ecosystem issue. Not seriously evaluated.
- **xArm Lite 6** (~$2,700). Better arm in every respect, real industrial drivers, official ROS2 support. Blew the budget.
- **Niryo Ned2** (~$2,000+). Educational arm. Blew the budget.
- **UR3e / Franka Research 3** (~$25K+). Industrial-grade. Way out of budget.
- **Seeed reBot Arm B601 / Trossen ViperX**. Open-source / research-grade but materially over budget for v1.

---

## 3. Existing Repo Investment That Influenced the Choice

By the time this decision was made, the repo had:

- **Phase 1 (vision pipeline) ✅ complete** — `board_detector.py`, `turn_tracker.py`, `ai_player.py`, `test_pipeline.py`, 8/8 on test images, AlphaZero ONNX inference working.
- **Phase 2 (live camera + cooperative game loop) ✅ complete** — `game_loop.py`, YOLOv8 fine-tuned on real captures, adaptive HSV, stable-state detector, TTS, full overlay.
- **Phase 3 (ROS2 simulation core) ✅ complete** — Ubuntu 24.04 + ROS2 Jazzy + Gazebo Harmonic + MoveIt2 + UR5e + `connect_four_arm` package reaching all 7 columns deterministically.
- **Phase 3B (VLA-ready sim) 🔄 in progress** — UR5e + Robotiq 2F-85 + pick-and-place sequence + overhead RGBD camera all integrated.

The JetArm path would have required abandoning the `connect_four_arm` ROS2 package's URDF/SRDF/MoveIt configs and rebuilding around HTS-21H gripper geometry. The SO-101 path keeps the simulation as a parallel research vehicle (UR5e for sim algorithm validation) and adds the SO-101 as the real-arm endpoint via the LeRobot toolchain.

---

## 4. Sources

- [Hiwonder JetArm product page](https://www.hiwonder.com/products/jetarm)
- [Hiwonder JetArm wiki — Getting Ready](https://wiki.hiwonder.com/projects/JetArm/en/jetarm-orin-nano/docs/1.Getting_Ready.html)
- [Hiwonder JetArm wiki — ROS2 Robot Arm Basic Control](https://wiki.hiwonder.com/projects/JetArm/en/jetarm-jetson-nano/docs/10.ROS2_Robot_Arm_Basic_ControlUser_Manual.html)
- [Hiwonder JetArm wiki — URDF Modeling & Simulation](https://docs.hiwonder.com/projects/JetArm/en/jetarm-orin-nano/docs/7.ROS1_Robot_Arm_URDF_Modeling_Simulation.html)
- [Hiwonder LeRobot SO-ARM101 product page](https://www.hiwonder.com/products/lerobot-so-101)
- [HuggingFace LeRobot — GitHub](https://github.com/huggingface/lerobot)
- [SO-ARM100 / SO-ARM101 hardware repo](https://github.com/TheRobotStudio/SO-ARM100)
- [Physical Intelligence](https://www.pi.website/) and [openpi](https://github.com/Physical-Intelligence/openpi) — informed the "stay in the LeRobot ecosystem" argument

---

## 5. Changelog

- **2026-04-28** — Decision finalized: SO-ARM101. Hiwonder support replies received (429 mm reach, 2–4 mm precision, Orin Nano Super "should be fully compatible"), confirming JetArm was technically capable but losing on ecosystem alignment and existing-investment grounds. Document renamed from `ROBOT_PLAN.md` to `ARM_DECISION_LOG.md` and reframed as a historical record.
