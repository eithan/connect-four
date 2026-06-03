#!/usr/bin/env bash
# =============================================================
#  view_episode.sh
#  Play back a recorded episode from the local LeRobot dataset.
#  Shows the front and hand camera feeds side by side.
#  No internet connection or Hugging Face upload required.
#
#  PREREQS (one-time):
#    sudo apt install ffmpeg      # provides ffplay
#
#  USAGE:
#    ./view_episode.sh            # play the most recent episode
#    ./view_episode.sh 3          # play episode 3 (zero-indexed)
#    ./view_episode.sh --list     # list all available episodes
#
#  The dataset used is set by REPO_ID below — change it to match
#  whichever dataset you want to inspect.
# =============================================================

set -euo pipefail

# ── Config ──────────────────────────────────────────────────────────────────
HF_USER="your-hf-username"
DATASET="connect_four_chute5_pick_col0"
REPO_ID="${HF_USER}/${DATASET}"

CACHE="${HOME}/.cache/huggingface/lerobot/${REPO_ID}"
VIDEO_DIR="${CACHE}/videos/chunk-000"
FRONT_DIR="${VIDEO_DIR}/observation.images.front"
HAND_DIR="${VIDEO_DIR}/observation.images.hand"

# ── Helpers ──────────────────────────────────────────────────────────────────
die() { echo "ERROR: $*" >&2; exit 1; }

check_dataset() {
    [[ -d "${CACHE}" ]] || die "Dataset not found at: ${CACHE}
  Run record_first_dataset.sh first, or update REPO_ID in this script."
    [[ -d "${FRONT_DIR}" ]] || die "No video files found at: ${FRONT_DIR}
  Make sure at least one episode has been recorded."
}

list_episodes() {
    check_dataset
    echo "Episodes in ${REPO_ID}:"
    echo ""
    local count=0
    for f in "${FRONT_DIR}"/episode_*.mp4; do
        [[ -f "$f" ]] || continue
        num=$(basename "$f" .mp4 | sed 's/episode_//')
        size=$(du -h "$f" | cut -f1)
        echo "  Episode $(( 10#$num ))   ${size}   $(basename "$f")"
        (( count++ )) || true
    done
    echo ""
    echo "Total: ${count} episode(s)"
}

play_episode() {
    local ep_num=$1
    local ep_str
    ep_str=$(printf "%06d" "$ep_num")

    local front="${FRONT_DIR}/episode_${ep_str}.mp4"
    local hand="${HAND_DIR}/episode_${ep_str}.mp4"

    [[ -f "$front" ]] || die "Episode ${ep_num} not found: ${front}"

    echo "Playing episode ${ep_num}..."
    echo "  Front camera: ${front}"
    [[ -f "$hand" ]] && echo "  Hand camera:  ${hand}"
    echo ""
    echo "Controls: space = pause/play,  q = quit,  left/right = seek"
    echo ""

    if [[ -f "$hand" ]]; then
        # Both cameras — show side by side with ffplay
        ffplay -hide_banner -loglevel warning \
            -window_title "Episode ${ep_num} — front | hand" \
            -f lavfi \
            "movie=${front},scale=640:480[v0]; \
             movie=${hand},scale=640:480[v1]; \
             [v0][v1]hstack" \
            2>/dev/null \
        || {
            # ffplay lavfi fallback failed — try simpler approach
            echo "(side-by-side failed, playing front then hand separately)"
            ffplay -hide_banner -loglevel warning \
                -window_title "Episode ${ep_num} — FRONT camera" "$front"
            ffplay -hide_banner -loglevel warning \
                -window_title "Episode ${ep_num} — HAND camera" "$hand"
        }
    else
        # Front camera only
        ffplay -hide_banner -loglevel warning \
            -window_title "Episode ${ep_num} — front camera" "$front"
    fi
}

get_latest_episode() {
    local latest=-1
    for f in "${FRONT_DIR}"/episode_*.mp4; do
        [[ -f "$f" ]] || continue
        num=$(basename "$f" .mp4 | sed 's/episode_//')
        num=$(( 10#$num ))
        (( num > latest )) && latest=$num
    done
    echo "$latest"
}

# ── Main ─────────────────────────────────────────────────────────────────────
check_dataset

ARG="${1:---latest}"

case "$ARG" in
    --list|-l)
        list_episodes
        ;;
    --latest|-n)
        ep=$(get_latest_episode)
        [[ $ep -ge 0 ]] || die "No episodes found in dataset."
        play_episode "$ep"
        ;;
    ''|*[!0-9]*)
        echo "Usage: $0 [EPISODE_NUMBER | --list | --latest]"
        exit 1
        ;;
    *)
        play_episode "$ARG"
        ;;
esac
