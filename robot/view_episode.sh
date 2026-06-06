#!/usr/bin/env bash
# =============================================================
#  view_episode.sh — GLOBAL SCRUB + OVERLAY (FIXED + CLEAN)
#
#  FEATURES:
#  - Correct timestamp-based episode alignment
#  - Global "scrub bar" across all episodes
#  - Episode index overlay on video
#  - No drift across MP4 boundaries
# =============================================================

set -euo pipefail

# ── Args ─────────────────────────────────────────────────────
COL=3
EPISODE_ARG=""
LIST_MODE=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --col|-c)
            shift; COL="$1"; shift ;;
        --list|-l)
            LIST_MODE=true; shift ;;
        --latest)
            EPISODE_ARG="latest"; shift ;;
        [0-9]*)
            EPISODE_ARG="$1"; shift ;;
        *)
            echo "Usage: $0 [--col N] [EPISODE | --list | --latest]"
            exit 1 ;;
    esac
done

# ── Config ───────────────────────────────────────────────────
HF_USER="eithanz"
REPO_ID="${HF_USER}/connect_four_chute5_pick_col${COL}"
DATASET_ROOT="${HOME}/lerobot_datasets/${REPO_ID}"

VIDEO_ROOT="${DATASET_ROOT}/videos"
FRONT="${VIDEO_ROOT}/observation.images.front/chunk-000/file-000.mp4"
HAND="${VIDEO_ROOT}/observation.images.hand/chunk-000/file-000.mp4"

die() { echo "ERROR: $*" >&2; exit 1; }

check_dataset() {
    [[ -d "$DATASET_ROOT" ]] || die "Dataset not found: $DATASET_ROOT"
}

count_episodes() {
python3 - <<PY
import json
d=json.load(open("${DATASET_ROOT}/meta/info.json"))
print(d.get("total_episodes", 0))
PY
}

# ─────────────────────────────────────────────────────────────
# GLOBAL TIMELINE (SCRUB BAR)
# ─────────────────────────────────────────────────────────────
build_timeline() {
python3 - <<PY
import pandas as pd
import pathlib

root = pathlib.Path("${DATASET_ROOT}")
df = pd.read_parquet(root / "data/chunk-000/file-000.parquet")

episodes = []
for ep in sorted(df["episode_index"].unique()):
    ep_df = df[df["episode_index"] == ep]

    start = float(ep_df["timestamp"].iloc[0])
    end   = float(ep_df["timestamp"].iloc[-1])

    episodes.append((ep, start, end, end - start))

# normalize so scrub starts at 0
t0 = episodes[0][1]
for e, s, t, d in episodes:
    print(e, s - t0, t - t0, d)
PY
}

# ─────────────────────────────────────────────────────────────
# EPISODE RANGE (timestamp-based, no drift)
# ─────────────────────────────────────────────────────────────
get_episode_range() {
python3 - <<PY
import pandas as pd
import pathlib

root = pathlib.Path("${DATASET_ROOT}")
ep = int("${1}")

df = pd.read_parquet(root / "data/chunk-000/file-000.parquet")
ep_df = df[df["episode_index"] == ep]

start = float(ep_df["timestamp"].iloc[0])
end   = float(ep_df["timestamp"].iloc[-1])

print(start, end)
PY
}

# ─────────────────────────────────────────────────────────────
# LIST MODE (SCRUB VIEW)
# ─────────────────────────────────────────────────────────────
list_episodes() {
    check_dataset

    echo "================================================"
    echo " GLOBAL SCRUB TIMELINE"
    echo "================================================"
    echo

    build_timeline | awk '
    {
        printf "Episode %-2s | %8.3fs → %8.3fs | dur %6.3fs\n",
               $1, $2, $3, $4
    }'
}

# ─────────────────────────────────────────────────────────────
# PLAY EPISODE WITH OVERLAY
# ─────────────────────────────────────────────────────────────
play_episode() {
    local ep="$1"

    read start end < <(
python3 - <<PY
import pandas as pd
import pathlib

root = pathlib.Path("${DATASET_ROOT}")
ep = int("${ep}")

df = pd.read_parquet(root / "data/chunk-000/file-000.parquet")
ep_df = df[df["episode_index"] == ep]

start = float(ep_df["timestamp"].iloc[0])
end   = float(ep_df["timestamp"].iloc[-1])

print(start, end)
PY
)

    dur=$(python3 -c "print(${end} - ${start})")

    echo "Playing Episode $ep"
    echo "Start: $start"
    echo "End:   $end"
    echo "Dur:   $dur"

    TEXT="Episode ${ep} | Col ${COL}"

    # FRONT camera (correct seek)
    ffplay -hide_banner -loglevel warning \
        -ss "$start" \
        -t "$dur" \
        -vf "drawtext=text='${TEXT}':x=10:y=10:fontsize=24:fontcolor=white:box=1:boxcolor=black@0.5" \
        -window_title "Front | Ep ${ep}" \
        "$FRONT" &

    PID1=$!

    # HAND camera (same timeline, same seek)
    ffplay -hide_banner -loglevel warning \
        -ss "$start" \
        -t "$dur" \
        -window_title "Hand | Ep ${ep}" \
        "$HAND" &

    PID2=$!

    wait $PID1 $PID2
}

# ─────────────────────────────────────────────────────────────
# LATEST EPISODE
# ─────────────────────────────────────────────────────────────
get_latest_episode() {
python3 - <<PY
import json
d=json.load(open("${DATASET_ROOT}/meta/info.json"))
print(d.get("total_episodes",0)-1)
PY
}

# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
check_dataset

if $LIST_MODE; then
    list_episodes

elif [[ "$EPISODE_ARG" == "" || "$EPISODE_ARG" == "latest" ]]; then
    ep=$(get_latest_episode)
    play_episode "$ep"

else
    play_episode "$EPISODE_ARG"
fi