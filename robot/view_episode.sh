#!/usr/bin/env bash
# =============================================================
#  view_episode.sh
#  Play back a recorded episode from the local LeRobot dataset.
#  Shows front and hand camera feeds side by side via ffplay.
#
#  PREREQS (one-time):
#    sudo apt install ffmpeg      # provides ffplay
#
#  USAGE:
#    ./view_episode.sh                  # col 3 (default), most recent episode
#    ./view_episode.sh --col 0          # col 0, most recent episode
#    ./view_episode.sh --col 0 5        # col 0, episode 5 (zero-indexed)
#    ./view_episode.sh --list           # col 3, list all episodes
#    ./view_episode.sh --col 0 --list   # col 0, list all episodes
#    ./view_episode.sh 5                # col 3, episode 5
# =============================================================

set -euo pipefail

# ── Args ──────────────────────────────────────────────────────────────────────
COL=3           # default column — change with --col N
EPISODE_ARG=""
LIST_MODE=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --col|-c)
            shift
            COL="$1"
            shift
            ;;
        --list|-l)
            LIST_MODE=true
            shift
            ;;
        --latest)
            EPISODE_ARG="--latest"
            shift
            ;;
        ''|*[0-9]*)
            EPISODE_ARG="$1"
            shift
            ;;
        *)
            echo "Usage: $0 [--col N] [EPISODE_NUMBER | --list | --latest]"
            exit 1
            ;;
    esac
done

# ── Config ────────────────────────────────────────────────────────────────────
HF_USER="eithanz"
DATASET="connect_four_chute5_pick_col${COL}"
REPO_ID="${HF_USER}/${DATASET}"
DATASET_ROOT="${HOME}/lerobot_datasets/${REPO_ID}"

# lerobot v0.5 stores videos as:
#   videos/observation.images.<cam>/chunk-<NNN>/file-<NNN>.mp4
# Episodes are split into chunks of CHUNK_SIZE. For ≤50 episodes everything
# is in chunk-000.
CHUNK_SIZE=1000
VIDEO_ROOT="${DATASET_ROOT}/videos"

# ── Helpers ───────────────────────────────────────────────────────────────────
die() { echo "ERROR: $*" >&2; exit 1; }

episode_to_path() {
    local cam=$1 ep=$2
    local chunk file
    chunk=$(printf "%03d" $(( ep / CHUNK_SIZE )))
    file=$(printf "%03d" $(( ep % CHUNK_SIZE )))
    echo "${VIDEO_ROOT}/observation.images.${cam}/chunk-${chunk}/file-${file}.mp4"
}

check_dataset() {
    [[ -d "${DATASET_ROOT}" ]] || die "Dataset not found at: ${DATASET_ROOT}
  Run record_first_dataset.sh first."
    [[ -d "${VIDEO_ROOT}" ]] || die "No videos directory found at: ${VIDEO_ROOT}
  Make sure at least one episode has been fully recorded."
}

count_episodes() {
    # Count unique file indices across all chunks for the front camera
    local cam_dir="${VIDEO_ROOT}/observation.images.front"
    [[ -d "$cam_dir" ]] || { echo 0; return; }
    find "$cam_dir" -name "file-*.mp4" | wc -l
}

list_episodes() {
    check_dataset
    local total
    total=$(count_episodes)
    echo "Dataset: ${REPO_ID}"
    echo "Root:    ${DATASET_ROOT}"
    echo "Episodes: ${total}"
    echo ""
    if [[ $total -eq 0 ]]; then
        echo "  (no episodes recorded yet)"
        return
    fi
    local i=0
    while [[ $i -lt $total ]]; do
        local front
        front=$(episode_to_path front "$i")
        local size="?"
        [[ -f "$front" ]] && size=$(du -h "$front" | cut -f1)
        echo "  Episode ${i}   ${size}   $(basename "$(dirname "$front")")/$(basename "$front")"
        (( i++ )) || true
    done
}

play_episode() {
    local ep_num=$1
    local front hand
    front=$(episode_to_path front "$ep_num")
    hand=$(episode_to_path hand "$ep_num")

    [[ -f "$front" ]] || die "Episode ${ep_num} not found: ${front}"

    echo "Playing episode ${ep_num}..."
    echo "  Front: ${front}"
    [[ -f "$hand" ]] && echo "  Hand:  ${hand}"
    echo ""
    echo "Controls: space=pause/play  q=quit  left/right=seek"
    echo ""

    if [[ -f "$hand" ]]; then
        # Side-by-side with ffplay's lavfi filter
        ffplay -hide_banner -loglevel warning \
            -window_title "Episode ${ep_num} — front | hand" \
            -f lavfi \
            "movie=${front},scale=640:480[v0]; \
             movie=${hand},scale=640:480[v1]; \
             [v0][v1]hstack" \
        2>/dev/null \
        || {
            echo "(side-by-side failed, playing cameras separately)"
            ffplay -hide_banner -loglevel warning \
                -window_title "Episode ${ep_num} — FRONT" "$front"
            ffplay -hide_banner -loglevel warning \
                -window_title "Episode ${ep_num} — HAND" "$hand"
        }
    else
        ffplay -hide_banner -loglevel warning \
            -window_title "Episode ${ep_num} — front" "$front"
    fi
}

get_latest_episode() {
    local total
    total=$(count_episodes)
    if [[ $total -eq 0 ]]; then
        echo -1
    else
        echo $(( total - 1 ))
    fi
}

# ── Main ──────────────────────────────────────────────────────────────────────
check_dataset

if $LIST_MODE; then
    list_episodes
elif [[ -z "$EPISODE_ARG" || "$EPISODE_ARG" == "--latest" ]]; then
    ep=$(get_latest_episode)
    [[ $ep -ge 0 ]] || die "No episodes found in dataset."
    play_episode "$ep"
else
    play_episode "$EPISODE_ARG"
fi
