#!/usr/bin/env bash
# =============================================================
#  record_first_dataset.sh
#  Record the v1 Connect Four dataset — 5-piece chute, depth-adaptive.
#
#  FIRST-TIME SETUP
#  ----------------
#  Install TTS engine (one-time, Ubuntu):
#    sudo apt install espeak
#
#  PREREQS EACH SESSION
#  ---------------------
#  - Arms plugged in and calibrated; teleop already verified.
#  - 5-piece chute at its marked spot, open sides facing the gripper axis.
#  - Connect Four board placed so column 0 is reachable.
#  - 5 discs loaded into the chute.
#
#  KEYBOARD CONTROLS
#  -------------------------------------------------------------------------
#  *** Wayland / RDP users: arrow keys won't work in the recording terminal.
#      Open a SECOND terminal and run:  python3 robot/send_key.py
#      Press ENTER there to inject a Right Arrow at the kernel level (evdev).
#      For Left Arrow (discard), see send_key.py — add a 'd' path if needed.
#
#  Normal X11 / local terminal:
#    Right Arrow : during setup  → start the episode early
#                  during record → end the episode early (only after returning
#                                  arm to start position)
#    Left  Arrow : discard current episode and re-record it
#    Ctrl-C      : stop the session cleanly (all completed episodes are kept)
#
#  EPISODE WORKFLOW
#  ----------------
#  The voice announcer tells you which phase you're in at all times.
#
#  SETUP TIME  →  position arm at start pose, press Right Arrow to begin
#  RECORDING   →  perform full task:
#                   1. Descend over chute, fingers either side of disc
#                   2. Close gripper on disc rim
#                   3. Lift disc clear of chute
#                   4. Rotate wrist 90° so disc is vertical
#                   5. Move to column 0
#                   6. Open gripper — disc falls in
#                   7. *** RETURN ARM TO START POSITION ***  ← do this every time
#                   8. Press Right Arrow to end episode
#  SETUP TIME  →  reload chute if announced, then press Right Arrow again
#
#  CHUTE RELOAD SCHEDULE  (every 5 episodes)
#  -------------------------------------------
#    Episodes  1– 5 : load at start, let deplete naturally
#    Episodes  6–10 : reload to 5 discs at start of episode 6
#    Episodes 11–15 : reload at start of episode 11
#    ...and so on. The voice will tell you when to reload.
#
#  STOPPING EARLY
#  ---------------
#  Press Ctrl-C. All completed episodes are saved automatically.
#  Run the script again to continue — it resumes from where you left off.
# =============================================================

set -euo pipefail

# ── Edit these ────────────────────────────────────────────────────────────────
HF_USER="eithanz"	# used for the dataset repo ID; push stays local
PUSH_TO_HUB="false"     # set "true" only when you want to upload to HF Hub

REPO_ID="${HF_USER}/connect_four_chute5_pick_col0"
# Local root for the dataset. --resume=true requires an explicit root (can't
# write into the Hub snapshot cache). Episodes accumulate here across sessions.
# Full path to this specific dataset — lerobot uses root as-is, no repo_id appended.
DATASET_ROOT="${HOME}/lerobot_datasets/${REPO_ID}"
TASK="Pick a yellow piece from the chute and drop it into column 0"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Kill Rerun on exit (Ctrl-C or normal finish) ──────────────────────────────
cleanup() {
    pkill -f "rerun" 2>/dev/null || true
}
trap cleanup EXIT

# ── Resume detection ─────────────────────────────────────────────────────────
# --resume=true triggers a Hub API call even when push_to_hub=false, which
# fails if the repo doesn't exist on HuggingFace. Only pass it when the local
# dataset directory already exists (i.e., this is a continuation, not a first run).
DATASET_LOCAL_PATH="${DATASET_ROOT}/${REPO_ID}"
RESUME_ARGS=()
if [ -d "${DATASET_LOCAL_PATH}" ]; then
    echo "Found existing dataset at ${DATASET_LOCAL_PATH} — resuming."
    RESUME_ARGS=(--resume=true)
else
    echo "No existing dataset found — starting fresh at ${DATASET_LOCAL_PATH}."
fi

# ── Run ───────────────────────────────────────────────────────────────────────
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
  --dataset.episode_time_s=45 \
  --dataset.reset_time_s=20 \
  --dataset.root="${DATASET_ROOT}" \
  --dataset.push_to_hub="${PUSH_TO_HUB}" \
  "${RESUME_ARGS[@]}" \
  --display_data=true \
  2>&1 | python3 "${SCRIPT_DIR}/record_voice_monitor.py"

# After recording, visualize the dataset before training:
#   python lerobot/scripts/visualize_dataset.py \
#     --repo-id your-username/connect_four_chute5_pick_col0
#
# Then copy to M4 Max and run train_act.sh with DEVICE="mps".
