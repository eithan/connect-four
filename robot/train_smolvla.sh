#!/usr/bin/env bash
# =============================================================
#  train_smolvla.sh
#  Fine-tune SmolVLA on the Connect Four pick-and-place dataset.
#
#  Run this on the M4 Max (DEVICE=mps). The Ubuntu machine's GPU
#  is too small for SmolVLA fine-tuning.
#
#  WORKFLOW
#  --------
#  1. On Ubuntu: finish recording, confirm episodes look good:
#       ./view_episode.sh --list
#  2. Copy dataset to Mac:
#       rsync -av ubuntu-host:~/lerobot_datasets/ ~/lerobot_datasets/
#  3. On Mac: run this script.
#  4. Copy checkpoint back to Ubuntu for eval:
#       rsync -av outputs/train/smolvla_c4_col0/ ubuntu-host:~/development/connect-four/robot/outputs/train/smolvla_c4_col0/
#  5. On Ubuntu: run the eval block at the bottom of this script.
#
#  FIRST-TIME SETUP (Mac, one-time)
#  ----------------------------------
#  - lerobot installed: cd ~/lerobot && pip install -e ".[feetech]"
#  - Apply the local-dataset patch (same as Ubuntu):
#      python3 robot/patch_lerobot.py   # see below
#  - Base model downloads automatically on first run (~1.5GB, needs internet once)
#
#  BASE MODEL
#  ----------
#  SmolVLA is fine-tuned FROM lerobot/smolvla_base (HuggingFace).
#  It downloads once and is cached locally. After the first run you
#  can train offline. The base model is pre-trained on ~100k robot
#  demonstrations across many tasks — you're adapting it to Connect Four.
#
#  WHY FEWER STEPS THAN ACT
#  -------------------------
#  ACT trains from scratch; 100k steps on 50 demos is needed.
#  SmolVLA is fine-tuning a pre-trained model — it already knows how
#  to grasp and manipulate. 10-20k steps on 10-20 demos is typically
#  enough. Start with 20k; check eval; extend if needed.
#
#  NOTE: flag names match LeRobot v0.5.x. Run `lerobot-train --help`
#  if any flag errors — the API drifts between versions.
# =============================================================

set -euo pipefail

# ── Match these to record_first_dataset.sh ───────────────────────────────────
HF_USER="eithanz"
REPO_ID="${HF_USER}/connect_four_chute5_pick_col3"
# Full path to the dataset — lerobot uses root as-is (no repo_id appended).
# Update this if your Mac dataset path differs.
DATASET_ROOT="${HOME}/lerobot_datasets/${REPO_ID}"

# ── Run config ────────────────────────────────────────────────────────────────
DEVICE="mps"                          # "mps" on Mac | "cuda" on Linux GPU
JOB_NAME="smolvla_c4_col3"
OUTPUT_DIR="outputs/train/${JOB_NAME}"

lerobot-train \
  --dataset.repo_id="${REPO_ID}" \
  --dataset.root="${DATASET_ROOT}" \
  --policy.type=smolvla \
  --policy.device="${DEVICE}" \
  --policy.push_to_hub=false \
  --output_dir="${OUTPUT_DIR}" \
  --job_name="${JOB_NAME}" \
  --batch_size=4 \
  --steps=20000 \
  --save_freq=5000 \
  --log_freq=100 \
  --num_workers=0 \
  --wandb.enable=false

# Checkpoints land in: ${OUTPUT_DIR}/checkpoints/<step>/pretrained_model
#
# Eval success guide:
#   < 40% — check trajectory consistency in recordings; may need more demos
#   40-60% — reasonable baseline; try more steps or more demos
#   60%+   — good; proceed to multi-column dataset and combined policy
#
# To extend training with more demos later:
#   --resume=true --output_dir="${OUTPUT_DIR}"  (same output dir)


# =============================================================
#  DEPLOY / EVAL ON THE REAL ARM  (run on Ubuntu after copying checkpoint)
# -------------------------------------------------------------------------
#  Copy checkpoint from Mac first:
#    rsync -av outputs/train/smolvla_c4_col0/ \
#      ubuntu-host:~/development/connect-four/robot/outputs/train/smolvla_c4_col0/
#
#  Then on Ubuntu:
#
#  CKPT="outputs/train/smolvla_c4_col0/checkpoints/last/pretrained_model"
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
#    --dataset.root="${HOME}/lerobot_datasets/${HF_USER}/eval_${JOB_NAME}" \
#    --dataset.single_task="Pick a yellow piece from the chute and drop it into column 3" \
#    --dataset.num_episodes=10 \
#    --dataset.episode_time_s=30 \
#    --dataset.reset_time_s=10 \
#    --dataset.push_to_hub=false \
#    --display_data=false
#
#  SAFETY: keep a hand near the power switch on the first policy rollout.
#  Camera keys (front/hand), device paths, and physical setup must match
#  recording exactly — any mismatch is out-of-distribution for the policy.
# =============================================================
