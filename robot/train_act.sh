#!/usr/bin/env bash
# =============================================================
#  train_act.sh — ACT training on M4 Max (MPS)
#
#  RUN:
#    cd ~/development/cursor/connect-four/robot
#    ./train_act.sh           # fresh run
#    ./train_act.sh --resume  # resume from last checkpoint
#
#  OUTPUT:
#    Checkpoints → outputs/train/act_c4_col3/checkpoints/
#    Progress bar + ETA printed every 100 steps.
#    Expected duration on M4 Max: ~21 hours for 50k steps.
#
#  NOTES:
#  - PYTORCH_ENABLE_MPS_FALLBACK=1 lets ops unsupported by MPS fall
#    back to CPU silently instead of crashing. Do not remove.
#  - PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 lets MPS use all GPU memory
#    instead of the default 50% cap.
#  - num_workers=0 avoids MPS multiprocessing hangs on Mac.
#  - 50k steps is appropriate for 25 episodes. If loss plateaus early,
#    Ctrl-C is safe — checkpoint is always saved before the next one.
# =============================================================

set -euo pipefail

# ── Parse args ────────────────────────────────────────────────────────────────
RESUME=false
for arg in "$@"; do
  [[ "$arg" == "--resume" ]] && RESUME=true
done

# ── Prevent Mac from sleeping during training ──────────────────────────────────
caffeinate -i &
CAFFEINATE_PID=$!
trap "kill $CAFFEINATE_PID 2>/dev/null" EXIT

# ── Config ────────────────────────────────────────────────────────────────────
HF_USER="eithanz"
REPO_ID="${HF_USER}/c4_col3_25"
DATASET_ROOT="${HOME}/.cache/huggingface/lerobot/${REPO_ID}"

DEVICE="mps"
JOB_NAME="act_c4_col3"
OUTPUT_DIR="outputs/train/${JOB_NAME}"
TOTAL_STEPS=50000

# ── MPS setup ─────────────────────────────────────────────────────────────────
export PYTORCH_ENABLE_MPS_FALLBACK=1
export KMP_DUPLICATE_LIB_OK=TRUE
# Allow MPS to use all available GPU memory (default cap is 50%).
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

# ── Train ─────────────────────────────────────────────────────────────────────
echo "Training ACT on ${REPO_ID}"
echo "  Device:  ${DEVICE}"
echo "  Steps:   ${TOTAL_STEPS}"
echo "  Output:  ${OUTPUT_DIR}"
echo "  Resume:  ${RESUME}"
echo ""

RESUME_FLAG=""
$RESUME && RESUME_FLAG="--resume=true"

lerobot-train \
  --dataset.repo_id="${REPO_ID}" \
  --dataset.root="${DATASET_ROOT}" \
  --policy.type=act \
  --policy.device="${DEVICE}" \
  --policy.push_to_hub=false \
  --output_dir="${OUTPUT_DIR}" \
  --job_name="${JOB_NAME}" \
  --batch_size=32 \
  --steps="${TOTAL_STEPS}" \
  --save_freq=5000 \
  --log_freq=100 \
  --num_workers=0 \
  --wandb.enable=false \
  ${RESUME_FLAG} \
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

    m = re.search(r'\bstep[:\s=]+([\d]+)(K?)', raw, re.IGNORECASE)
    if not m:
        continue
    step = int(m.group(1)) * (1000 if m.group(2).upper() == 'K' else 1)
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
    finish     = datetime.datetime.now() + datetime.timedelta(seconds=int(remaining))
    finish_str = finish.strftime('%H:%M')
    pct        = step / total
    filled     = int(BAR * pct)
    bar        = '█' * filled + '░' * (BAR - filled)
    last_shown = step

    print(f'  [{bar}] {100*pct:.1f}%  |  Done ~{finish_str}  |  {rate:.1f} steps/s', flush=True)
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
#  CKPT="$HOME/development/connect-four/robot/outputs/train/act_c4_col3/checkpoints/last/pretrained_model"
#
#  lerobot-rollout \
#    --robot.type=so101_follower \
#    --robot.port=/dev/so101_follower \
#    --robot.id=my_follower_arm \
#    --robot.cameras='{
#      front: {type: opencv, index_or_path: "/dev/video2", width: 640, height: 480, fps: 30, fourcc: "MJPG", backend: "V4L2"},
#      hand:  {type: opencv, index_or_path: "/dev/video0", width: 640, height: 480, fps: 30, fourcc: "MJPG", backend: "V4L2"}
#    }' \
#    --policy.type=act \
#    --policy.pretrained_path="${CKPT}" \
#    --device=cuda \
#    --fps=15 \
#    --task="Pick a yellow piece from the chute and drop it into column 3" \
#    --display_data=false \
#    --strategy.type=base
#
#  NOTES:
#  - --strategy.type=base: run policy only, no episode recording.
#  - --duration=0 (default): runs until Ctrl-C.
#  - On Ctrl-C, arm automatically returns to its start position.
#  - --fps=15 matches training data rate (critical for correct chunk timing).
#  - SAFETY: keep a hand near the power switch on the first policy rollout.
# =============================================================
