#!/usr/bin/env bash
# =============================================================
#  record_first_dataset.sh
#  Record the v1 Connect Four dataset — 5-piece chute, depth-adaptive.
#  See dataset_schema.md for the schema this implements.
#
#  GOAL
#  ----
#  Train ACT to pick from ALL 5 disc positions in the chute (full stack
#  down to last disc) without needing to reload between picks. The policy
#  learns to go deeper as pieces disappear by seeing the visual change
#  in the wrist camera.
#
#  PREREQS
#  -------
#  - Arms calibrated; teleop already verified (your lerobot-teleoperate cmd).
#  - 5-piece chute placed at its marked spot on the bench (tape cross).
#    Orient so the ±X open faces align with the gripper's open/close axis.
#  - Board placed so column 0 is reachable.
#  - You are logged in to the HF Hub if pushing:  huggingface-cli login
#
#  RECORDING WORKFLOW — IMPORTANT
#  --------------------------------
#  DO NOT reload the chute between episodes within a cycle.
#  Let the chute deplete naturally so the dataset covers all 5 depths:
#
#    Cycle start  →  load 5 discs into chute
#    Episode  1   →  pick disc 1 (top of full stack), drop into column 0
#    Episode  2   →  pick disc 2 (one deeper), drop into column 0
#    Episode  3   →  pick disc 3, drop
#    Episode  4   →  pick disc 4, drop
#    Episode  5   →  pick disc 5 (bottom, deepest), drop
#    Cycle end    →  reload chute to full (5 discs), start next cycle
#
#  With 50 episodes = 10 full cycles = 10 demos per disc depth.
#  That gives ACT enough visual coverage to interpolate across all depths.
#
#  During the reset_time_s window between episodes:
#    - If the chute still has discs: just return arm to start pose. Done.
#    - If the chute is now empty (every 5th episode): reload it fully,
#      then return arm to start pose. You have 15 s.
#
#  KEYBOARD CONTROLS (focus on the terminal/Rerun window)
#  -------------------------------------------------------
#    Right Arrow : end current episode early and move on
#    Left  Arrow : discard and re-record the current episode
#                  (re-seat the disc you just picked if needed)
#    Escape      : stop the whole session
#
#  NOTE: flag names match LeRobot v0.5.x. If a flag errors, run
#        `lerobot-record --help` — the API does drift between versions.
# =============================================================

set -euo pipefail

# ---- Edit these ----
HF_USER="your-hf-username"          # <-- set your Hugging Face username
PUSH_TO_HUB="false"                 # "true" to upload; keep local while testing

REPO_ID="${HF_USER}/connect_four_chute5_pick_col0"
TASK="Pick a yellow piece from the chute and drop it into column 0"

lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/so101_follower \
  --robot.id=my_follower_arm \
  --robot.cameras='{
    front: {type: opencv, index_or_path: "/dev/video0", width: 640, height: 480, fps: 30, fourcc: "MJPG", backend: "V4L2"},
    hand:  {type: opencv, index_or_path: "/dev/video2", width: 640, height: 480, fps: 30, fourcc: "YUYV", backend: "V4L2"}
  }' \
  --teleop.type=so101_leader \
  --teleop.port=/dev/so101_leader \
  --teleop.id=my_leader_arm \
  --dataset.repo_id="${REPO_ID}" \
  --dataset.single_task="${TASK}" \
  --dataset.fps=30 \
  --dataset.num_episodes=50 \
  --dataset.episode_time_s=20 \
  --dataset.reset_time_s=15 \
  --dataset.push_to_hub="${PUSH_TO_HUB}" \
  --display_data=true

# After this completes, the dataset lives under your LeRobot home
# (~/.cache/huggingface/lerobot/${REPO_ID} by default).
#
# Visualize before training:
#   python lerobot/scripts/visualize_dataset.py --repo-id ${REPO_ID}
#
# Then run train_act.sh (remember to set DEVICE="mps" for M4 Max).
