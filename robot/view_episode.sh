#!/usr/bin/env bash
# =============================================================
#  view_episode.sh
#  Play back a recorded episode from the local LeRobot dataset.
#  Shows front and hand camera feeds side by side.
#
#  NOTE: lerobot v0.5 concatenates all episodes in a chunk into
#  a single video file (chunk-000/file-000.mp4). This script reads
#  the parquet data to find exact frame boundaries, then pipes
#  ffmpeg → ffplay to seek accurately.
#
#  PREREQS (one-time):
#    sudo apt install ffmpeg
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
COL=3
EPISODE_ARG=""
LIST_MODE=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --col|-c)  shift; COL="$1"; shift ;;
        --list|-l) LIST_MODE=true; shift ;;
        --latest)  EPISODE_ARG="--latest"; shift ;;
        [0-9]*)    EPISODE_ARG="$1"; shift ;;
        *)
            echo "Usage: $0 [--col N] [EPISODE_NUMBER | --list | --latest]"
            exit 1
            ;;
    esac
done

# ── Config ────────────────────────────────────────────────────────────────────
HF_USER="eithanz"
REPO_ID="${HF_USER}/connect_four_chute5_pick_col${COL}"
DATASET_ROOT="${HOME}/lerobot_datasets/${REPO_ID}"
VIDEO_ROOT="${DATASET_ROOT}/videos"
FRONT="${VIDEO_ROOT}/observation.images.front/chunk-000/file-000.mp4"
HAND="${VIDEO_ROOT}/observation.images.hand/chunk-000/file-000.mp4"

# ── Helpers ───────────────────────────────────────────────────────────────────
die() { echo "ERROR: $*" >&2; exit 1; }

check_dataset() {
    [[ -d "${DATASET_ROOT}" ]] || die "Dataset not found: ${DATASET_ROOT}
  Run: ./record_first_dataset.sh ${COL}"
    [[ -f "${FRONT}" ]] || die "No video at: ${FRONT}
  Record at least one episode first."
}

# Read episode count from metadata.
count_episodes() {
    local info="${DATASET_ROOT}/meta/info.json"
    [[ -f "$info" ]] || { echo 0; return; }
    python3 -c "import json; d=json.load(open('$info')); print(d.get('total_episodes', d.get('num_episodes', 0)))"
}

# Return "start_seconds duration_seconds" for episode N.
# lerobot v3.0 layout:
#   meta/episodes/chunk-{C:03d}/file-{F:03d}.parquet  — per-episode metadata (has 'length')
#   data/chunk-{C:03d}/file-{F:03d}.parquet           — per-frame data (fallback)
get_episode_timing() {
    local ep=$1
    python3 - "$DATASET_ROOT" "$ep" <<'PYEOF'
import sys, json, pathlib
import pandas as pd

root      = pathlib.Path(sys.argv[1])
ep_target = int(sys.argv[2])
fps       = float(json.loads((root / "meta" / "info.json").read_text()).get("fps", 15))

# ── Primary: meta/episodes parquet (compact, pre-computed lengths) ────────────
for f in sorted((root / "meta" / "episodes").glob("chunk-*/*.parquet")):
    df = pd.read_parquet(f)
    if "length" in df.columns and "episode_index" in df.columns:
        df = df.sort_values("episode_index").reset_index(drop=True)
        if ep_target < len(df):
            lengths     = df["length"].tolist()
            start_frame = sum(lengths[:ep_target])
            ep_length   = lengths[ep_target]
            print(f"{start_frame/fps:.3f} {ep_length/fps:.3f}")
            sys.exit(0)

# ── Fallback: data parquet (one row per frame) ────────────────────────────────
for f in sorted((root / "data").glob("chunk-*/*.parquet")):
    df = pd.read_parquet(f, columns=["episode_index"])
    # lerobot sometimes stores scalar fields as length-1 arrays
    ep_col = df["episode_index"]
    if ep_col.dtype == object:
        ep_col = ep_col.apply(lambda x: int(x[0]) if hasattr(x, "__len__") else int(x))
        df["episode_index"] = ep_col
    counts      = df.groupby("episode_index").size()
    start_frame = int(sum(counts.get(i, 0) for i in range(ep_target)))
    ep_length   = int(counts.get(ep_target, 0))
    if ep_length > 0:
        print(f"{start_frame/fps:.3f} {ep_length/fps:.3f}")
        sys.exit(0)

print("ERROR: could not determine episode timing", file=sys.stderr)
sys.exit(1)
PYEOF
}

list_episodes() {
    check_dataset
    local total; total=$(count_episodes)
    echo "Dataset:  ${REPO_ID}"
    echo "Root:     ${DATASET_ROOT}"
    echo "Episodes: ${total}"
    echo ""
    [[ $total -gt 0 ]] || { echo "  (no episodes recorded yet)"; return; }

    local front_size; front_size=$(du -h "$FRONT" | cut -f1)
    echo "  Video: ${FRONT}"
    echo "  Size:  ${front_size}  (all ${total} episodes concatenated)"
    echo ""
    printf "  %7s  %10s  %12s\n" "Episode" "Start (s)" "Duration (s)"
    printf "  %7s  %10s  %12s\n" "-------" "---------" "------------"

    local i=0
    while [[ $i -lt $total ]]; do
        local timing
        if timing=$(get_episode_timing "$i" 2>/dev/null); then
            local start dur
            read -r start dur <<< "$timing"
            printf "  %7d  %10.1f  %12.1f\n" "$i" "$start" "$dur"
        else
            printf "  %7d  %10s  %12s\n" "$i" "?" "?"
        fi
        (( i++ )) || true
    done
}

play_episode() {
    local ep_num=$1
    local total; total=$(count_episodes)
    [[ $ep_num -lt $total ]] || die "Episode ${ep_num} doesn't exist (dataset has ${total} episodes, 0-indexed)."

    local timing
    timing=$(get_episode_timing "$ep_num") || die "Could not read timing for episode ${ep_num}."
    local start_s dur_s
    read -r start_s dur_s <<< "$timing"

    local end_s
    end_s=$(python3 -c "print(${start_s}+${dur_s})")

    echo "Playing col ${COL} episode ${ep_num}/${total}..."
    printf "  Seek: %.1fs   Duration: %.1fs\n" "$start_s" "$dur_s"
    echo ""
    echo "Controls: space=pause/play  q=quit  left/right=seek within this clip"
    echo ""

    if [[ -f "$HAND" ]]; then
        # Use ffmpeg to decode both cameras to the right window, pipe to ffplay.
        # -ss before -i = fast keyframe seek; -t = duration.
        # Pipe through NUT container to ffplay (no temp files).
        ffmpeg -hide_banner -loglevel warning \
            -ss "$start_s" -t "$dur_s" -i "$FRONT" \
            -ss "$start_s" -t "$dur_s" -i "$HAND" \
            -filter_complex \
              "[0:v]scale=640:480,setpts=PTS-STARTPTS[v0]; \
               [1:v]scale=640:480,setpts=PTS-STARTPTS[v1]; \
               [v0][v1]hstack[out]" \
            -map "[out]" \
            -f nut pipe:1 2>/dev/null \
        | ffplay -hide_banner -loglevel warning \
            -window_title "Col ${COL} | Ep ${ep_num}/${total} — front | hand" \
            -
    else
        ffplay -hide_banner -loglevel warning \
            -window_title "Col ${COL} | Ep ${ep_num}/${total} — front" \
            -ss "$start_s" -t "$dur_s" \
            "$FRONT"
    fi
}

get_latest_episode() {
    local total; total=$(count_episodes)
    [[ $total -gt 0 ]] && echo $(( total - 1 )) || echo -1
}

# ── Main ──────────────────────────────────────────────────────────────────────
check_dataset

if $LIST_MODE; then
    list_episodes
elif [[ -z "$EPISODE_ARG" || "$EPISODE_ARG" == "--latest" ]]; then
    ep=$(get_latest_episode)
    [[ $ep -ge 0 ]] || die "No episodes found."
    play_episode "$ep"
else
    play_episode "$EPISODE_ARG"
fi
