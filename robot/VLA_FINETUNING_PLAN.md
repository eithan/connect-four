# VLA Fine-Tuning Plan — SO-101 + openpi/LeRobot

**Companion to:** [`PROJECT_PLAN.md`](./PROJECT_PLAN.md). This document is the deep-dive on Vision-Language-Action (VLA) fine-tuning that complements `PROJECT_PLAN.md` Phase 3C (SO-101 hardware setup) and Phase 4+ (full game integration). It captures the architectural patterns and tooling choices distilled from Physical Intelligence's openpi and HuggingFace's LeRobot.

For arm selection rationale see [`ARM_DECISION_LOG.md`](./ARM_DECISION_LOG.md). For overall project status see [`SESSION_STATUS.md`](./SESSION_STATUS.md).

---

## 1. Why This Document Exists

`PROJECT_PLAN.md` already covers the SO-101 ordering, calibration, demo recording, and a vanilla ACT training step in Phase 3C. This document adds the layer above that:

- The **LeRobot dataset format** as the universal interchange standard (and why getting it right early matters)
- Three concrete VLA fine-tuning options — **SmolVLA**, **π₀-base**, **π₀.₅-base** — with deployment patterns for each
- The **remote inference + action chunking** architecture pattern from Physical Intelligence's openpi
- The **architectural lessons from PI** that improve the system regardless of which VLA you pick

If you're skipping VLA work (sticking with scripted motion or a simple ACT baseline), you can ignore this file entirely.

---

## 2. Architecture Lessons from Physical Intelligence

Physical Intelligence (the makers of π₀, π₀.₅, π₀.₇, π*0.6, openpi) has open-sourced their architecture and thinking. Their patterns inform this plan even if you don't run their models directly.

### 2.1 Separate policy from execution; stream action chunks

The openpi reference architecture is **remote inference**: the model runs on a powerful GPU server, the robot subscribes to a websocket, the server pushes chunks of ~50 actions at a time, and the robot executes them in sequence. This pattern:

- Decouples model size from on-robot compute (run a 5B-param VLA on a desktop GPU; Jetson just executes).
- Makes the robot platform substitutable (any LeRobot-compatible follower can subscribe).
- Maps cleanly onto our existing ROS2 action contract (see Phase 3C in `PROJECT_PLAN.md`): `arm_node` becomes a thin client subscribing to a policy server.

```
[GPU server with VLA]  ←── observations (cam frames, joint state) ── [Jetson]
                       ──── action chunk (50 actions) ──→
                                                                     [Jetson]
                                                                       │
                                                                       ▼ ROS2 action server
                                                                   arm_node
                                                                       │
                                                                       ▼ Feetech bus
                                                                   SO-101 follower
```

### 2.2 Action chunking is a smoothness pattern, not just a learned-policy pattern

Both π₀ (flow-matching) and π₀-FAST (autoregressive) emit **chunks** because predicting one action at a time is jittery and slow. The same idea applies to scripted motion in Phase 3C: generate a dense trajectory upfront with `ruckig` and stream it at 100 Hz, rather than sending discrete waypoints.

This is why the existing `column_mover.py` already does precomputed IK and joint-space planning — same idea, different label.

### 2.3 LeRobot dataset format is the de facto standard

Every published VLA fine-tuning recipe consumes LeRobot dataset format — SmolVLA, π₀, π₀.₅, OpenVLA, ACT, Diffusion Policy. Recording demos in any other format means a future migration project. **All Phase 3C demonstration recording must produce LeRobot-format datasets.**

### 2.4 Don't train from scratch — fine-tune a pre-trained foundation model

PI's whole thesis is that pre-trained foundation VLAs (trained on 10K+ hours of cross-platform robot data) are the right substrate, and you fine-tune them on your specific task with hundreds (not thousands) of demos. For Connect Four:

- **Don't** build a custom net.
- **Do** fine-tune SmolVLA / π₀-base / π₀.₅-base on your recorded demos.

Cheaper, faster, better.

### 2.5 Cross-embodiment generalization is the bet

PI's models work across 7+ robot types after fine-tuning on minutes-to-hours of new-platform data. The practical implication:

- **Don't over-invest in arm-specific code.** Anything in `vision_node`, `game_node`, `policy_node`, or the orchestration layer should remain platform-independent.
- **Only `arm_node` should know what arm it's driving.** The current `connect_four_arm` ROS2 package already follows this; preserve it.

This is also why the existing UR5e simulation continues to have value alongside the SO-101 real-arm work — algorithms validated on UR5e in sim transfer to SO-101 with minimal arm-specific changes.

### 2.6 Language conditioning is becoming free

π₀.₅ and π₀.₇ take natural-language commands ("clean the bedroom"). For multi-game support (tic-tac-toe, checkers, household tasks), accepting a language goal in the action interface is a low-cost addition. Worth considering an optional `prompt: string` field in the ROS2 action contract for v2.

### 2.7 They cloud-host their large models — and call it production-grade

A Jetson Orin Nano cannot run π₀.₇ (5B + 14B + 400M parameters) locally. It can run SmolVLA (450M). HuggingFace built SmolVLA specifically to be the on-device counterpart to PI's cloud models. **For on-Jetson inference, SmolVLA is the ceiling, not π₀ or larger.**

---

## 3. Three VLA Fine-Tuning Options

In order of capability, compute requirement, and deployment complexity. Pick one or sequence them.

### Option A — SmolVLA (recommended starting point)

| Aspect | Value |
|---|---|
| Parameters | ~450M |
| Designed for | SO-100 / SO-101 specifically (HuggingFace LeRobot) |
| On-Jetson inference | Yes — fits in 8 GB Orin Nano memory |
| Training | Single GPU; ~hours on RTX 3080+ |
| Maturity | Production-ready as of LeRobot v0.5.0+ |

**Why start here:** Most realistic on-device target. Designed for exactly the hardware you're buying. Direct continuation of the LeRobot ACT path described in `PROJECT_PLAN.md` Phase 3C — same dataset, same training script, different policy class.

**Deployment shape:** All-on-Jetson. Policy server and `arm_node` co-located. No remote inference infrastructure needed.

### Option B — π₀-base (Physical Intelligence open-source)

| Aspect | Value |
|---|---|
| Parameters | ~3B |
| Pre-training data | 10K+ hours, 7+ robot platforms |
| On-Jetson inference | **No** — requires desktop GPU or cloud |
| Training | Single high-end GPU (4090/5090/A100) or rented cloud |
| Maturity | Production-ready in openpi |

**Why use it:** Better generalization than SmolVLA per PI's published numbers, especially with limited fine-tuning data. Strong language-following.

**Deployment shape:** **Remote inference required.** Run `openpi/scripts/serve_policy.py` on a desktop GPU server; Jetson runs `arm_node` as a websocket client streaming action chunks to the SO-101 follower.

### Option C — π₀.₅-base

| Aspect | Value |
|---|---|
| Parameters | Larger than π₀; semantic reasoning co-trained |
| Pre-training data | π₀ data plus heterogeneous web data and high-level task supervision |
| On-Jetson inference | **No** — desktop GPU or cloud |
| Training | Single high-end GPU; meaningfully heavier than π₀ |
| Maturity | Production-ready in openpi |

**Why use it:** Strongest available open-weight VLA. Best open-world generalization (PI's mobile-manipulator-cleaning-a-new-house demos). Most useful for the long-term "household tasks" goal.

**Deployment shape:** Same as π₀-base — remote inference.

### Decision rule of thumb

- Want a self-contained Jetson demo? → **SmolVLA.**
- Have a desktop GPU and want best Connect Four performance? → **π₀-base.**
- Eyeing household-task generalization? → **π₀.₅-base.**

Sequencing them — start with SmolVLA as the on-Jetson baseline, then add π₀.₅-base on a desktop server for comparison — is the most-learning-per-week path.

---

## 4. Tasks (companion to `PROJECT_PLAN.md` Phase 3C)

The PROJECT_PLAN.md already covers the prerequisite hardware and demo-recording steps. These tasks layer on top of those.

### 4.1 Dataset hygiene (during Phase 3C demo recording)

- [ ] Verify LeRobot's `record` script writes datasets in **LeRobot format** (default — confirm version compatibility with both `lerobot` and `openpi` as of date of recording).
- [ ] Add **natural-language annotations** per episode (e.g., "place a red piece in column 3"). Costs nothing during recording; opens up VLA fine-tuning paths.
- [ ] Push dataset to **HuggingFace Hub** (private repo) — both for portability between training environments and for the dataset visualizer.
- [ ] Document the dataset's input/output schema in a `dataset_card.md` in the dataset repo.

### 4.2 SmolVLA fine-tune (Option A)

- [ ] Confirm LeRobot v0.5.0+ environment on training GPU.
- [ ] Fine-tune SmolVLA on the Connect Four dataset.
- [ ] Deploy weights to Jetson; benchmark inference latency at 30–50 Hz.
- [ ] Build `arm_node_smolvla` (variant of `arm_node`) that runs SmolVLA inference and streams action chunks to the SO-101.
- [ ] Compare task success and motion quality against the scripted baseline from PROJECT_PLAN.md Phase 3C.4 ACT.

### 4.3 π₀-base or π₀.₅-base fine-tune (Option B/C)

- [ ] Set up the **GPU server** (desktop with 4090/5090, or rented cloud).
- [ ] Install openpi (`Physical-Intelligence/openpi`), pull base checkpoints from `gs://openpi-assets/checkpoints/`.
- [ ] Convert / verify the C.1 dataset is in **openpi-compatible LeRobot format** (openpi consumes LeRobot datasets directly).
- [ ] Fine-tune the chosen base model on Connect Four data.
- [ ] Stand up policy server: `uv run scripts/serve_policy.py policy:checkpoint --policy.config=<config> --policy.dir=<checkpoint_dir>`
- [ ] Build `arm_node_openpi` that subscribes to the policy server via websocket and streams action chunks to the SO-101 follower.
- [ ] Add network latency / disconnect handling — fall back to home pose on policy server failure.
- [ ] Benchmark task success, motion quality, end-to-end latency vs. SmolVLA baseline.

### 4.4 Stretch experiments

- [ ] **Zero-shot transfer probe.** Test the fine-tuned policy on small variations: different piece colors, shifted board position, slightly different lighting. Measures whether the foundation-model pre-training is paying off.
- [ ] **Multi-task generalization.** Add a second task (e.g., tic-tac-toe pick-and-place) to the dataset with a different language prompt; check whether the policy can switch tasks based on prompt.
- [ ] **Public dataset release.** Open-source the Connect Four dataset on HuggingFace Hub as a contribution to the LeRobot community — documents the SO-101 + Connect Four use case for future work.

---

## 5. Architectural Diagram — v2 (learned policy)

When the v1 scripted-motion `arm_node` is replaced by a learned policy, only `arm_node` changes. Everything upstream is unchanged.

### 5.1 SmolVLA (on-Jetson)

```
┌──────────────────────────────────────────────────────────────────┐
│                 Jetson Orin Nano 8GB Super (Ubuntu 24.04 + ROS2 Jazzy) │
│                                                                  │
│  vision_node ──▶ game_manager_node ──▶ policy_node               │
│                          │                  │                    │
│                          │                  ▼                    │
│                          │           AlphaZero ONNX              │
│                          ▼                  │                    │
│                 /connect_four/ai_move (col N)                    │
│                          │                                       │
│                          ▼                                       │
│                  arm_node_smolvla                                │
│                  (SmolVLA inference + action streaming)          │
│                          │                                       │
└──────────────────────────┼───────────────────────────────────────┘
                           │ Feetech serial bus
                           ▼
                       SO-101 follower
```

### 5.2 π₀-base / π₀.₅-base (remote inference)

```
[Desktop / cloud GPU server]                      [Jetson Orin Nano]
┌──────────────────────────────┐                  ┌──────────────────────────┐
│  openpi serve_policy         │                  │  vision_node             │
│  ┌────────────────────────┐  │  observations    │  game_manager_node       │
│  │ π₀-base / π₀.₅-base    │◀─┼──────────────────┤  arm_node_openpi         │
│  │ (fine-tuned on C4 data)│  │                  │  (websocket client)      │
│  └─────────────┬──────────┘  │                  │                          │
│                │             │                  │                          │
│                ▼             │                  │                          │
│  Action chunk (50 actions)   │  action chunk    │                          │
│                              │──────────────────▶│                         │
└──────────────────────────────┘                  │      │                   │
                                                  └──────┼───────────────────┘
                                                         │ Feetech bus
                                                         ▼
                                                    SO-101 follower
```

The websocket policy server is openpi's reference deployment — same pattern PI uses for production Weave/Ultra deployments.

---

## 6. References

### Physical Intelligence
- [Physical Intelligence — main site](https://www.pi.website/)
- [π₀: Our First Generalist Policy](https://www.pi.website/blog/pi0)
- [π₀.₅: A VLA with Open-World Generalization](https://www.pi.website/blog/pi05)
- [π*0.6: A VLA that Learns from Experience](https://www.pi.website/blog/pistar06)
- [π₀.₇: A Steerable Model with Emergent Capabilities](https://www.pi.website/blog/pi07)
- [Open Sourcing π₀](https://www.pi.website/blog/openpi)
- [openpi — GitHub](https://github.com/Physical-Intelligence/openpi)
- [openpi remote inference docs](https://github.com/Physical-Intelligence/openpi/blob/main/docs/remote_inference.md)

### LeRobot ecosystem
- [HuggingFace LeRobot — GitHub](https://github.com/huggingface/lerobot)
- [LeRobot v0.5.0 release notes](https://huggingface.co/blog/lerobot-release-v050)
- [SmolVLA blog](https://huggingface.co/blog/smolvla)
- [SmolVLA model card](https://huggingface.co/lerobot/smolvla_base)
- [pi0_base on HuggingFace](https://huggingface.co/lerobot/pi0_base)
- [pi05_base on HuggingFace](https://huggingface.co/lerobot/pi05_base)
- [SO-ARM100 / SO-ARM101 hardware repo](https://github.com/TheRobotStudio/SO-ARM100)

### Supporting
- [`ruckig` — jerk-limited trajectory generation](https://github.com/pantor/ruckig) (relevant for action-chunk streaming)
- [DROID dataset](https://droid-dataset.github.io/)
- [LIBERO benchmark](https://libero-project.github.io/datasets)

---

## 7. Changelog

- **2026-04-29** — Document renamed from `ROBOT_PLAN_ALT.md` to `VLA_FINETUNING_PLAN.md` and reframed as a focused companion to `PROJECT_PLAN.md` rather than a competing plan. Removed sections that duplicated `PROJECT_PLAN.md` (general SO-101 setup, calibration, basic demo recording). Added explicit Physical Intelligence architecture lessons section, three VLA fine-tuning options with deployment patterns, and v2 architecture diagrams for both on-Jetson and remote-inference deployments.
- **2026-04-28** — Initial version (as `ROBOT_PLAN_ALT.md`). Three-phase plan with Phase A heavy on fabrication and LeRobot environment setup; Phase C centered on imitation learning + SmolVLA fine-tuning as the path's payoff.
