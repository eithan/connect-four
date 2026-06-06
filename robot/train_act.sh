#!/usr/bin/env bash
# =============================================================
#  train_act.sh
#  Train an ACT policy on the v1 Connect Four dataset
#  (fixed pick -> column 0). Run this the moment recording finishes.
#  See dataset_schema.md for what the dataset contains.
#
#  PREREQS
#  -------
#  - Dataset recorded via record_first_dataset.sh (same REPO_ID below).
#    lerobot-train reads it from the local LeRobot cache by repo_id;
#    if you pushed to the Hub it will pull from there instead.
#  - A CUDA GPU on this machine is strongly preferred. Check first:
#        nvidia-smi
#    No discrete GPU? Set DEVICE below to "cpu" (slow) or "mps" (Apple).
#    This was an open question in SESSION_STATUS.md — confirm before a
#    long run so you don't discover it 6 hours in.
#
#  RUN
#  ---
#    chmod +x train_act.sh
#    ./train_act.sh
#
#  NOTE: flag names match LeRobot v0.5.x. If a flag errors, run
#        `lerobot-train --help` — the API drifts between versions.
# =============================================================

set -euo pipefail

# ---- Match these to record_first_dataset.sh ----
HF_USER="eithanz"
REPO_ID="${HF_USER}/connect_four_chute5_pick_col3"
# Full path to the dataset — lerobot uses root as-is (no repo_id appended).
# If running on Mac after copying from Ubuntu, update this path accordingly.
DATASET_ROOT="${HOME}/lerobot_datasets/${REPO_ID}"

# ---- Run config ----
DEVICE="mps"                          # "mps" on Mac | "cuda" on Linux GPU
JOB_NAME="act_c4_col3"
OUTPUT_DIR="outputs/train/${JOB_NAME}"

lerobot-train \
  --dataset.repo_id="${REPO_ID}" \
  --dataset.root="${DATASET_ROOT}" \
  --policy.type=act \
  --policy.device="${DEVICE}" \
  --policy.push_to_hub=false \
  --output_dir="${OUTPUT_DIR}" \
  --job_name="${JOB_NAME}" \
  --batch_size=8 \
  --steps=100000 \
  --save_freq=10000 \
  --log_freq=200 \
  --num_workers=4 \
  --wandb.enable=false

# Checkpoints land in:  ${OUTPUT_DIR}/checkpoints/<step>/pretrained_model
# and a rolling          ${OUTPUT_DIR}/checkpoints/last/pretrained_model
#
# WHY 100k steps + save_freq=10k: for a 50-episode single-task set, ACT
# usually converges well before 100k. Saving every 10k lets you pick the
# best checkpoint by eval instead of guessing the step count up front.
# To add more demos later and continue:  --resume=true with the same
# --output_dir (see `lerobot-train --help`).


# =============================================================
#  DEPLOY / EVAL ON THE REAL ARM  (run AFTER training — not part of
#  this script; copy-paste when you have a checkpoint)
# -------------------------------------------------------------
#  This is the same lerobot-record entrypoint, but the POLICY drives the
#  follower (no leader/teleop). It records evaluation episodes so you can
#  measure task success against the scripted baseline.
#
#  CKPT="${OUTPUT_DIR}/checkpoints/last/pretrained_model"
#
#  lerobot-record \
#    --robot.type=so101_follower \
#    --robot.port=/dev/so101_follower \
#    --robot.id=my_follower_arm \
#    --robot.cameras='{
#      front: {type: opencv, index_or_path: "/dev/video2", width: 640, height: 480, fps: 30, fourcc: "MJPG", backend: "V4L2"},
#      hand:  {type: opencv, index_or_path: "/dev/video0", width: 640, height: 480, fps: 30, fourcc: "MJPG", backend: "V4L2"}
#    }' \
#    --policy.path="${CKPT}" \
#    --dataset.repo_id="${HF_USER}/eval_${JOB_NAME}" \
#    --dataset.single_task="Pick a yellow piece from the chute and drop it into column 3" \
#    --dataset.num_episodes=10 \
#    --dataset.episode_time_s=20 \
#    --dataset.reset_time_s=10 \
#    --dataset.push_to_hub=false \
#    --display_data=true
#
#  SAFETY: keep a hand near the e-stop / power on the first policy rollout.
#  The cameras + their KEYS (front, hand) and the disc/board setup must
#  match recording exactly, or the policy sees an out-of-distribution scene.
# =============================================================
