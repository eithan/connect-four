#!/usr/bin/env bash
# =============================================================
#  view_episode.sh
#  Play back a recorded episode from the local LeRobot dataset.
#  Shows front and hand camera feeds side by side via ffplay.
#
#  NOTE: lerobot v0.5 stores all episodes in a chunk concatenated
#  into a single video file (chunk-000/file-000.mp4). This script
#  reads episode metadata to seek to the correct position within
#  that file.
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
        [0-9]*)
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

# lerobot v0.5 concatenates all episodes in a chunk into one file.
# All episodes (< 1000) land in chunk-000/file-000.mp4.
VIDEO_ROOT="${DATASET_ROOT}/videos"
VIDEO_FILE_FRONT="${VIDEO_ROOT}/observation.images.front/chunk-000/file-000.mp4"
VIDEO_FILE_HAND="${VIDEO_ROOT}/observation.images.hand/chunk-000/file-000.mp4"

# ── Helpers ───────────────────────────────────────────────────────────────────
die() { echo "ERROR: $*" >&2; exit 1; }

check_dataset() {
    [[ -d "${DATASET_ROOT}" ]] || die "Dataset not found at: ${DATASET_ROOT}
  Run: ./record_first_dataset.sh ${COL}"
    [[ -f "${VIDEO_FILE_FRONT}" ]] || die "No video found at: ${VIDEO_FILE_FRONT}
  Make sure at least one episode has been fully recorded."
}

# Read total episode count from lerobot's metadata (info.json).
# Falls back to counting data parquet files if needed.
count_episodes() {
    local info="${DATASET_ROOT}/meta/info.json"
    if [[ -f "$info" ]]; then
        python3 -c "import json; d=json.load(open('$info')); print(d.get('total_episodes', d.get('num_episodes', 0)))"
    else
        echo 0
    fi
}

# Return "start_seconds duration_seconds" for a given episode index.
# Reads cumulative frame counts from meta/episodes.jsonl (lerobot v0.5).
get_episode_timing() {
    local ep=$1
    python3 - "$DATASET_ROOT" "$ep" <<'PYEOF'
import sys, json, pathlib

root = pathlib.Path(sys.argv[1])
ep   = int(sys.argv[2])

info = json.loads((root / "meta" / "info.json").read_text())
fps  = float(info.get("fps", 15))

# Try episodes.jsonl (lerobot v0.5 standard)
eps_file = root / "meta" / "episodes.jsonl"
if eps_file.exists():
    lines = [l for l in eps_file.read_text().strip().split("\n") if l]
    episodes = [json.loads(l) for l in lines]
    if ep >= len(episodes):
        print(f"0 0", file=sys.stderr)
        sys.exit(1)
    lengths = [e.get("length", 0) for e in episodes]
    start_frame  = sum(lengths[:ep])
    ep_length    = lengths[ep]
else:
    # Fallback: assume equal-length episodes
    total_frames = info.get("total_frames", 0)
    total_ep     = info.get("total_episodes", 1)
    ep_length    = total_frames // total_ep
    start_frame  = ep * ep_length

print(f"{start_frame/fps:.3f} {ep_length/fps:.3f}")
PYEOF
}

list_episodes() {
    check_dataset
    local total
    total=$(count_episodes)
    echo "Dataset:  ${REPO_ID}"
    echo "Root:     ${DATASET_ROOT}"
    echo "Episodes: ${total}"
    echo "Video:    ${VIDEO_FILE_FRONT}"
    echo ""
    if [[ $total -eq 0 ]]; then
        echo "  (no episodes recorded yet)"
        return
    fi
    local front_size
    front_size=$(du -h "$VIDEO_FILE_FRONT" | cut -f1)
    echo "  Front video: ${front_size}  (all ${total} episodes concatenated)"
    echo ""
    echo "  Episode  Start(s)   Duration(s)"
    echo "  ──────────────────────────────"
    local i=0
    while [[ $i -lt $total ]]; do
        local timing
        timing=$(get_episode_timing "$i" 2>/dev/null) || { echo "  Episode ${i}   (metadata error)"; (( i++ )) || true; continue; }
        local start dur
        read -r start dur <<< "$timing"
        printf "  %7d  %8.1f   %10.1f\n" "$i" "$start" "$dur"
        (( i++ )) || true
    done
}

play_episode() {
    local ep_num=$1
    local total
    total=$(count_episodes)
    [[ $ep_num -lt $total ]] || die "Episode ${ep_num} does not exist (dataset has ${total} episodes, 0-indexed)."

    local timing
    timing=$(get_episode_timing "$ep_num") || die "Could not read timing metadata for episode ${ep_num}."
    local start_s dur_s
    read -r start_s dur_s <<< "$timing"

    echo "Playing episode ${ep_num} of ${total}..."
    echo "  Seek:     ${start_s}s    Duration: ${dur_s}s"
    echo "  Front:    ${VIDEO_FILE_FRONT}"
    [[ -f "$VIDEO_FILE_HAND" ]] && echo "  Hand:     ${VIDEO_FILE_HAND}"
    echo ""
    echo "Controls: space=pause/play  q=quit  left/right=seek"
    echo ""

    if [[ -f "$VIDEO_FILE_HAND" ]]; then
        # Side-by-side: seek both cameras to the same episode window
        ffplay -hide_banner -loglevel warning \
            -window_title "Col ${COL} | Episode ${ep_num}/${total} — front | hand" \
            -f lavfi \
            "movie=${VIDEO_FILE_FRONT}:seek_point=${start_s},trim=duration=${dur_s},scale=640:480[v0]; \
             movie=${VIDEO_FILE_HAND}:seek_point=${start_s},trim=duration=${dur_s},scale=640:480[v1]; \
             [v0][v1]hstack" \
        2>/dev/null \
        || {
            echo "(side-by-side failed, playing front camera only)"
            ffplay -hide_banner -loglevel warning \
                -window_title "Col ${COL} | Episode ${ep_num}/${total} — FRONT" \
                -ss "$start_s" -t "$dur_s" \
                "$VIDEO_FILE_FRONT"
        }
    else
        ffplay -hide_banner -loglevel warning \
            -window_title "Col ${COL} | Episode ${ep_num}/${total} — front" \
            -ss "$start_s" -t "$dur_s" \
            "$VIDEO_FILE_FRONT"
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
