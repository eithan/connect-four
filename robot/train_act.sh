#!/usr/bin/env bash
# =============================================================
#  train_act.sh — ACT training on M4 Max (MPS)
#
#  BEFORE RUNNING (one-time on Mac):
#    rsync -av --progress eithan@<ubuntu-ip>:~/lerobot_datasets/ ~/lerobot_datasets/
#
#  RUN:
#    cd ~/development/cursor/connect-four/robot
#    ./train_act.sh
#
#  OUTPUT:
#    Checkpoints → outputs/train/act_c4_col3/checkpoints/
#    Progress bar + ETA printed every 100 steps.
#    Expected duration on M4 Max: 3–6 hours for 50k steps.
#
#  NOTES:
#  - PYTORCH_ENABLE_MPS_FALLBACK=1 lets ops unsupported by MPS fall
#    back to CPU silently instead of crashing.
#  - num_workers=0 avoids MPS multiprocessing hangs.
#  - 50k steps is appropriate for 10 episodes; 100k risks overfitting.
#    If loss plateaus early (watch the ETA bar), Ctrl-C is safe —
#    the last checkpoint is always saved before the next one starts.
# =============================================================

set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
HF_USER="eithanz"
REPO_ID="${HF_USER}/connect_four_chute5_pick_col3"
DATASET_ROOT="${HOME}/lerobot_datasets/${REPO_ID}"

DEVICE="mps"
JOB_NAME="act_c4_col3"
OUTPUT_DIR="outputs/train/${JOB_NAME}"
TOTAL_STEPS=50000

# ── MPS setup ─────────────────────────────────────────────────────────────────
export PYTORCH_ENABLE_MPS_FALLBACK=1

# ── Train ─────────────────────────────────────────────────────────────────────
echo "Training ACT on ${REPO_ID}"
echo "  Device:  ${DEVICE}"
echo "  Steps:   ${TOTAL_STEPS}"
echo "  Output:  ${OUTPUT_DIR}"
echo ""

lerobot-train \
  --dataset.repo_id="${REPO_ID}" \
  --dataset.root="${DATASET_ROOT}" \
  --policy.type=act \
  --policy.device="${DEVICE}" \
  --policy.push_to_hub=false \
  --output_dir="${OUTPUT_DIR}" \
  --job_name="${JOB_NAME}" \
  --batch_size=16 \
  --steps="${TOTAL_STEPS}" \
  --save_freq=5000 \
  --log_freq=100 \
  --num_workers=4 \
  --wandb.enable=false \
2>&1 | python3 -u -c "
import sys, re, time, datetime

total       = ${TOTAL_STEPS}
start_time  = None
first_step  = None
last_shown  = -1
BAR         = 30

for raw in sys.stdin:
    sys.stdout.write(raw)
    sys.stdout.flush()

    m = re.search(r'\bstep[:\s=]+(\d+)', raw, re.IGNORECASE)
    if not m:
        continue
    step = int(m.group(1))
    if step == 0 or step <= last_shown:
        continue

    now = time.time()
    if start_time is None:
        start_time = now
        first_step = step
        continue

    elapsed    = now - start_time
    steps_done = step - first_step
    if steps_done <= 0:
        continue

    rate       = steps_done / elapsed
    remaining  = (total - step) / rate if rate > 0 else 0
    eta        = datetime.timedelta(seconds=int(remaining))
    pct        = step / total
    filled     = int(BAR * pct)
    bar        = '█' * filled + '░' * (BAR - filled)
    last_shown = step

    print(f'  [{bar}] {100*pct:.1f}%  |  ETA {eta}  |  {rate:.1f} steps/s', flush=True)
"

echo ""
echo "Training complete. Checkpoints in: ${OUTPUT_DIR}/checkpoints/"


# =============================================================
#  DEPLOY / EVAL ON THE REAL ARM  (run on Ubuntu after copying checkpoint)
# -------------------------------------------------------------
#
#  1. Copy checkpoint from Mac to Ubuntu:
#       rsync -av outputs/train/act_c4_col3/ \
#         eithan@<ubuntu-ip>:~/development/connect-four/robot/outputs/train/act_c4_col3/
#
#  2. On Ubuntu, run:
#
#  CKPT="outputs/train/act_c4_col3/checkpoints/last/pretrained_model"
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
# =============================================================
