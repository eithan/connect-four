#!/usr/bin/env bash
# =============================================================
#  record_first_dataset.sh  [COLUMN]
#  Record a Connect Four pick-and-place dataset for a given column.
#
#  USAGE
#  -----
#    ./record_first_dataset.sh       # column 0, 20 episodes (first recording)
#    ./record_first_dataset.sh 3     # column 3, 10 episodes
#    ./record_first_dataset.sh 0 5   # column 0, 5 episodes (override count)
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
#  - Connect Four board placed so target column is reachable.
#  - 5 discs loaded into the chute.
#
#  KEYBOARD CONTROLS
#  -------------------------------------------------------------------------
#  *** Wayland / RDP users: arrow keys won't work in the recording terminal.
#      lerobot uses pynput which listens on X11 (DISPLAY env var).
#      1. sudo apt install xdotool       (one-time)
#      2. Open a SECOND terminal and run: python3 robot/send_key.py
#         ENTER = Right Arrow, d+ENTER = Left Arrow
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
#                   5. Move to target column
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
#  Wait until you see "Recording episode N" (not "Reset the environment") before
#  pressing Ctrl-C. lerobot encodes the video during the reset phase — killing
#  it mid-encode discards the episode. Once the next episode number appears,
#  encoding of the previous one is done and Ctrl-C is safe.
#  Run the script again to continue — it resumes from where you left off.
# =============================================================

set -euo pipefail

# ── Args ──────────────────────────────────────────────────────────────────────
COL="${1:-0}"
if ! [[ "${COL}" =~ ^[0-6]$ ]]; then
    echo "Usage: $0 [COLUMN (0-6)] [NUM_EPISODES]" >&2
    exit 1
fi

# Column 0 gets 20 episodes (pipeline validation); all others get 10.
DEFAULT_EPISODES=$([ "${COL}" -eq 0 ] && echo 20 || echo 10)
NUM_EPISODES="${2:-${DEFAULT_EPISODES}}"

# ── Config ────────────────────────────────────────────────────────────────────
HF_USER="eithanz"
PUSH_TO_HUB="false"

REPO_ID="${HF_USER}/connect_four_chute5_pick_col${COL}"
DATASET_BASE="${HOME}/lerobot_datasets"
TASK="Pick a yellow piece from the chute and drop it into column ${COL}"

echo "Column: ${COL}  |  Episodes: ${NUM_EPISODES}  |  Dataset: ${REPO_ID}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Kill Rerun on exit (Ctrl-C or normal finish) ──────────────────────────────
cleanup() {
    pkill -f "rerun" 2>/dev/null || true
}
trap cleanup EXIT

# ── Resume detection ─────────────────────────────────────────────────────────
# Both create() and resume() use --dataset.root as the full dataset path (no
# repo_id appended internally). Always pass the full path.
#
# --resume=true also triggers a Hub API call even with push_to_hub=false.
# Apply the one-time lerobot patch (see PREREQS above) to fix this.
DATASET_LOCAL_PATH="${DATASET_BASE}/${REPO_ID}"
DATASET_ROOT="${DATASET_LOCAL_PATH}"
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
    front: {type: opencv, index_or_path: "/dev/video2", width: 640, height: 480, fps: 30, fourcc: "MJPG", backend: "V4L2"},
    hand:  {type: opencv, index_or_path: "/dev/video0", width: 640, height: 480, fps: 30, fourcc: "MJPG", backend: "V4L2"}
  }' \
  --teleop.type=so101_leader \
  --teleop.port=/dev/so101_leader \
  --teleop.id=my_leader_arm \
  --dataset.repo_id="${REPO_ID}" \
  --dataset.single_task="${TASK}" \
  --dataset.fps=15 \
  --dataset.num_episodes="${NUM_EPISODES}" \
  --dataset.episode_time_s=30 \
  --dataset.reset_time_s=8 \
  --dataset.root="${DATASET_ROOT}" \
  --dataset.push_to_hub="${PUSH_TO_HUB}" \
  "${RESUME_ARGS[@]}" \
  --display_data=false

# After recording, visualize the dataset before training:
#   python lerobot/scripts/visualize_dataset.py \
#     --repo-id your-username/connect_four_chute5_pick_col0
#
# Then copy to M4 Max and run train_act.sh with DEVICE="mps".
